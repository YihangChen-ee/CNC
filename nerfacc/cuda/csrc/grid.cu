#include <torch/extension.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/util/MaybeOwned.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

#include "include/data_spec.hpp"
#include "include/data_spec_packed.cuh"
#include "include/utils_cuda.cuh"
#include "include/utils_grid.cuh"
#include "include/utils_math.cuh"

static constexpr uint32_t MAX_GRID_LEVELS = 8;

namespace {
namespace device {

inline __device__ float _calc_dt(
    const float t, const float cone_angle,
    const float dt_min, const float dt_max)
{
    return clamp(t * cone_angle, dt_min, dt_max);
}

/* Ray traversal within multiple voxel grids. 

About rays:
    Each ray is defined by its origin (rays_o) and unit direction (rays_d). We also allows
    a optional boolen ray mask (rays_mask) to indicate whether we want to skip some rays. 

About voxel grids:
    We support ray traversal through one or more voxel grids (n_grids). Each grid is defined
    by an axis-aligned AABB (aabbs), and a binary occupancy grid (binaries) with resolution of
    {resx, resy, resz}. Currently, we assume all grids have the same resolution. Note the ordering
    of the grids is important when there are overlapping grids, because we assume the grid in front
    has higher priority when examing occupancy status (e.g., the first grid's occupancy status
    will overwrite the second grid's occupancy status if they overlap).

About ray grid intersections:
    We require the ray grid intersections to be precomputed and sorted. Specifically, if hit, each 
    ray-grid pair has two intersections, one for entering the grid and one for leaving the grid. 
    For multiple grids, there are in total 2 * n_grids intersections for each ray. The intersections
    are sorted by the distance to the ray origin (t_sorted). We take a boolen array (hits) to indicate 
    whether each ray-grid pair is hit. We also need a int64 array (t_indices) to indicate the grid id
    (0-index) for each intersection.

About ray traversal:
    The ray is traversed through the grids in the order of the sorted intersections. We allows pre-ray
    near and far planes (near_planes, far_planes) to be specified. Early termination can be controlled by
    setting the maximum traverse steps via traverse_steps_limit. We also allow an optional step size
    (step_size) to be specified. If step_size <= 0.0, we will record the steps of the ray pass through
    each voxel cell. Otherwise, we will use the step_size to march through the grids. When step_size > 0.0,
    we also allow a cone angle (cone_angle) to be provides, to linearly increase the step size as the ray
    goes further away from the origin (see _calc_dt()). cone_angle should be always >= 0.0, and 0.0 
    means uniform marching with step_size.

About outputs:
    The traversal intervals and samples are stored in `intervals` and `samples` respectively. Additionally,
    we also return where the traversal actually terminates (terminate_planes). This is useful when 
    traverse_steps_limit is set (traverse_steps_limit > 0) as the ray may not reach the far plane or the
    boundary of the grids.
*/
__global__ void traverse_grids_kernel(
    // rays
    int32_t n_rays,
    float *rays_o,  // [n_rays, 3]
    float *rays_d,  // [n_rays, 3]
    bool *rays_mask, // [n_rays]
    // grids
    int32_t n_grids,
    int3 resolution,
    bool *binaries, // [n_grids, resx, resy, resz]
    float *aabbs,   // [n_grids, 6] [[-1.5000, -1.5000, -1.5000,  1.5000,  1.5000,  1.5000]]
    // sorted intersections
    bool *hits,         // [n_rays, n_grids]
    float *t_sorted,    // [n_rays, n_grids * 2]  // 这里代表的是xyz中的最快轴
    int64_t *t_indices, // [n_rays, n_grids * 2]
    // options
    float *near_planes,  // [n_rays] 对于blender，是0
    float *far_planes,   // [n_rays] 对于blender，是1e10
    float step_size,  // 默认是1e-3
    float cone_angle,
    int32_t traverse_steps_limit,
    // outputs
    bool first_pass,
    PackedRaySegmentsSpec intervals,
    PackedRaySegmentsSpec samples,
    float *terminate_planes)
{
    float eps = 1e-6f;

    // parallelize over rays
    for (int32_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n_rays; tid += blockDim.x * gridDim.x)
    {
        if (rays_mask != nullptr && !rays_mask[tid]) continue;

        // skip rays that are empty.
        if (intervals.chunk_cnts != nullptr)
            if (!first_pass && intervals.chunk_cnts[tid] == 0) continue;
        if (samples.chunk_cnts != nullptr)
            if (!first_pass && samples.chunk_cnts[tid] == 0) continue;

        int64_t chunk_start, chunk_start_bin;
        if (!first_pass) {
            if (intervals.chunk_cnts != nullptr)
                chunk_start = intervals.chunk_starts[tid];
            if (samples.chunk_cnts != nullptr)
                chunk_start_bin = samples.chunk_starts[tid];
        }
        float near_plane = near_planes[tid];
        float far_plane = far_planes[tid];

        // 这边是指针。*3是因为3维。对每一条光线进行遍历。
        SingleRaySpec ray = SingleRaySpec(
            rays_o + tid * 3, rays_d + tid * 3, near_plane, far_plane);

        int32_t base_hits = tid * n_grids;
        // 变量 base_t_sorted 表示用于访问 t_sorted 数组的特定射线的起始索引。
        // t_sorted 数组包含每条射线与每个网格的排序交点距离，其大小为 [n_rays, n_grids * 2]。每一行对应于一条特定的射线，每一对列表示沿射线路径进入和离开每个网格的交点距离。
        // 变量 base_t_sorted 的计算方式是 tid * n_grids * 2，其中 tid 表示线程ID。它作为访问特定射线的 t_sorted 数组的起始索引。
        // 通过使用 base_t_sorted，代码可以通过从 base_t_sorted 到 base_t_sorted + n_grids * 2 - 1 的索引范围内访问特定射线的交点距离。这样，代码就能够遍历特定射线的排序交点，并在遍历过程中进行相应的处理。
        int32_t base_t_sorted = tid * n_grids * 2;

        // loop over all intersections along the ray.
        // 这里开始循环遍历沿光线的交点，这些交点是按距离排序的。
        int64_t n_intervals = 0;
        int64_t n_samples = 0;
        float t_last = near_plane;
        // continuous: means whether current traverse point is continuous with both its previous and next point along the ray.
        // it will be false if 1. it is the first 
        bool continuous = false;
        // 对grid进行遍历。对于blender，只有一个grid
        for (int32_t i = base_t_sorted; i < base_t_sorted + n_grids * 2 - 1; i++) {
            // whether this is the entering or leaving for this level of grid.
            bool is_entering = t_indices[i] < n_grids;
            int64_t level = t_indices[i] % n_grids;
            // printf("i=%d, level=%lld, is_entering=%d, hits=%d\n", i, level, is_entering, hits[level]);

            // 在对每个grid的循环迭代中，检查当前ray是否在grid内，如果不在grid内，则跳过。
            if (!hits[base_hits + level]) {
                continue; // this grid is not hit.
            }

            // 如果当前交点不是进入网格的点（即，它是离开网格的点），则检查下一个交点是否进入另一个网格。如果是，说明光线在网格之间穿越，继续下一轮循环。
            if (!is_entering) {
                // we are leaving this grid. Are we inside the next grid?
                bool next_is_entering = t_indices[i + 1] < n_grids;
                if (next_is_entering) continue; // we are outside next grid.
                level = t_indices[i + 1] % n_grids;
                if (!hits[base_hits + level]) {
                    continue; // this grid is not hit.
                }
            }

            // 在每个循环迭代中，计算有效区间的最小和最大值。
            float this_tmin = fmaxf(t_sorted[i], near_plane);
            float this_tmax = fminf(t_sorted[i + 1], far_plane);   
            if (this_tmin >= this_tmax) continue; // this interval is invalid. e.g. (0.0f, 0.0f)
            // printf("i=%d, this_tmin=%f, this_tmax=%f, level=%lld\n", i, this_tmin, this_tmax, level);

            if (!continuous) {  // continuous: means whether current traverse point is continuous with both its previous and next point along the ray.
                if (step_size <= 0.0f) { // march to this_tmin.
                    t_last = this_tmin;
                } else {
                    // 对于blender，cone_angle=0，所以dt=step_size
                    float dt = _calc_dt(t_last, cone_angle, step_size, 1e10f);
                    while (true) { // march until t_mid is right after this_tmin.
                        if (t_last + dt * 0.5f >= this_tmin) break;
                        t_last += dt;  // 修改了t_last，为下一个xx做准备
                    }
                }
            }
            // printf(
            //     "[traverse segment] i=%d, this_mip=%d, this_tmin=%f, this_tmax=%f\n", 
            //     i, this_mip, this_tmin, this_tmax);

            AABBSpec aabb = AABBSpec(aabbs + level * 6);

            // init: pre-compute variables needed for traversal
            float3 tdist, delta;
            // tdist 中的每个分量（tdist.x、tdist.y 和 tdist.z）包含了光线在对应轴上到达下一个交点的距离信息。
            // delta 这个变量表示在体素网格中沿着光线方向的步进量
            int3 step_index, current_index, final_index;
            // get voxel index of start and end within grid
            setup_traversal(
                ray, this_tmin, this_tmax, eps,
                aabb, resolution,
                // outputs
                delta, tdist, step_index, current_index, final_index);
            // printf(
            //     "[traverse init], delta=(%f, %f, %f), step_index=(%d, %d, %d)\n",
            //     delta.x, delta.y, delta.z, step_index.x, step_index.y, step_index.z
            // );

            const int3 overflow_index = final_index + step_index;
            while (traverse_steps_limit <= 0 || n_samples < traverse_steps_limit) {
                // The code calculates t_traverse as the minimum value among the distances tdist.x, tdist.y, and tdist.z. 
                // These distances represent the distances to the next grid cell boundaries along the ray's path.
                float t_traverse = min(tdist.x, min(tdist.y, tdist.z)); // 和t_sorted一样，这里只考虑最快轴
                // The code then clamps t_traverse to ensure it does not exceed this_tmax, 
                // which represents the maximum intersection distance for the current grid cell.
                t_traverse = fminf(t_traverse, this_tmax);
                // resolution = make_int3(binaries.size(1), binaries.size(2), binaries.size(3));
                int64_t cell_id = (
                    current_index.x * resolution.y * resolution.z
                    + current_index.y * resolution.z
                    + current_index.z
                    + level * resolution.x * resolution.y * resolution.z
                );

                if (!binaries[cell_id]) {
                    // skip the cell that is empty.
                    if (step_size <= 0.0f) { // march to t_traverse.
                        t_last = t_traverse;
                    } else {
                        // 对于blender，cone_angle=0，所以dt=step_size
                        float dt = _calc_dt(t_last, cone_angle, step_size, 1e10f);
                        while (true) { // march until t_mid is right after t_traverse.
                            if (t_last + dt * 0.5f >= t_traverse) break;
                            t_last += dt;  // 修改了t_last，为下一个xx做准备
                        }
                    }
                    continuous = false;
                } else {
                    // this cell is not empty, so we need to traverse it.
                    // traverse_steps_limit: Maximum number of samples per ray. default: None
                    // 这个while表示在当前binary cell里进行工作。是否超出了当前cell是通过t_traverse来判断的。
                    // t_traverse是根据binary的resolution和rays_d得到的，在当前cell的最大遍历步长。
                    while (traverse_steps_limit <= 0 || n_samples < traverse_steps_limit) {
                        float t_next;
                        if (step_size <= 0.0f) {
                            t_next = t_traverse;
                        } else {  // march until t_mid is right after t_traverse.
                            float dt = _calc_dt(t_last, cone_angle, step_size, 1e10f);
                            if (t_last + dt * 0.5f >= t_traverse) break;  // 如果中点超出t_traverse了就跳出
                            t_next = t_last + dt;
                        }

                        /* 在循环中
                        idx:                    0   1   2   3
                        intervals.vals:         t0  t1  t2  t3
                        intervals.is_left:      T   T   T
                        intervals.is_right:         T   T   T
                        */

                        // writeout the interval.
                        if (intervals.chunk_cnts != nullptr) {
                            if (!continuous) {
                                if (!first_pass) {  // left side of the intervel
                                    int64_t idx = chunk_start + n_intervals;
                                    intervals.vals[idx] = t_last;  // 在外面的代码中会通过(t_origins + t_dirs * (t_starts[:, None] + t_ends[:, None]) / 2.0)转化成3D坐标
                                    intervals.ray_indices[idx] = tid;
                                    intervals.is_left[idx] = true;
                                }
                                n_intervals++;
                                if (!first_pass) {  // right side of the intervel
                                    int64_t idx = chunk_start + n_intervals;
                                    intervals.vals[idx] = t_next;  // 在外面的代码中会通过(t_origins + t_dirs * (t_starts[:, None] + t_ends[:, None]) / 2.0)转化成3D坐标
                                    intervals.ray_indices[idx] = tid;
                                    intervals.is_right[idx] = true;
                                }
                                n_intervals++;
                            } else {
                                if (!first_pass) {  // right side of the intervel
                                    int64_t idx = chunk_start + n_intervals;
                                    intervals.vals[idx] = t_next;  // 在外面的代码中会通过(t_origins + t_dirs * (t_starts[:, None] + t_ends[:, None]) / 2.0)转化成3D坐标
                                    intervals.ray_indices[idx] = tid;
                                    intervals.is_left[idx - 1] = true;
                                    intervals.is_right[idx] = true;
                                }
                                n_intervals++;
                            }
                        }

                        // writeout the sample.
                        if (samples.chunk_cnts != nullptr) {
                            if (!first_pass) {
                                int64_t idx = chunk_start_bin + n_samples;

                                samples.vals[idx] = (t_next + t_last) * 0.5f; // 实际计算的positions=(t_origins + t_dirs * (t_starts[:, None] + t_ends[:, None]) / 2.0)
                                // (t_next + t_last) * 0.5f = (t_last + dt + t_last) * 0.5f = t_last + dt*0.5 < t_traverse

                                samples.ray_indices[idx] = tid;
                                samples.is_valid[idx] = true;
                            }
                        }

                        n_samples++;
                        continuous = true;
                        t_last = t_next;
                        if (t_next >= t_traverse) break;
                    }
                }

                // printf(
                //     "[traverse], t_last=%f, t_traverse=%f, cell_id=%d, current_index=(%d, %d, %d)\n",
                //     t_last, t_traverse, cell_id, current_index.x, current_index.y, current_index.z
                // );

                if (!single_traversal(tdist, current_index, overflow_index, step_index, delta)) {
                    break;
                }
            }
        }
        if (terminate_planes != nullptr)
            terminate_planes[tid] = t_last;

        if (intervals.chunk_cnts != nullptr)
            intervals.chunk_cnts[tid] = n_intervals;
        if (samples.chunk_cnts != nullptr)
            samples.chunk_cnts[tid] = n_samples;
    }
}

__global__ void ray_aabb_intersect_kernel(
    const int32_t n_rays, float *rays_o, float *rays_d, float near, float far,
    const int32_t n_aabbs, float *aabbs,
    // outputs
    const float miss_value,
    float *t_mins, float *t_maxs, bool *hits)
{
    int32_t numel = n_rays * n_aabbs;
    // parallelize over rays
    for (int32_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < numel; tid += blockDim.x * gridDim.x)
    {
        int32_t ray_id = tid / n_aabbs;
        int32_t aabb_id = tid % n_aabbs;

        float t_min, t_max;
        bool hit = device::ray_aabb_intersect(
            SingleRaySpec(rays_o + ray_id * 3, rays_d + ray_id * 3, near, far), 
            AABBSpec(aabbs + aabb_id * 6), 
            t_min, t_max
        );
        if (hit) {   
            t_mins[tid] = t_min;
            t_maxs[tid] = t_max;
        } else {
            t_mins[tid] = miss_value;
            t_maxs[tid] = miss_value;
        }
        hits[tid] = hit;
    }
}


}  // namespace device
}  // namespace


std::tuple<RaySegmentsSpec, RaySegmentsSpec, torch::Tensor> traverse_grids(
    // rays
    const torch::Tensor rays_o, // [n_rays, 3]
    const torch::Tensor rays_d, // [n_rays, 3]
    const torch::Tensor rays_mask,   // [n_rays]
    // grids
    const torch::Tensor binaries,  // [n_grids, resx, resy, resz]
    const torch::Tensor aabbs,     // [n_grids, 6]  [[-1.5000, -1.5000, -1.5000,  1.5000,  1.5000,  1.5000]]
    // intersections
    const torch::Tensor t_sorted,  // [n_rays, n_grids]
    const torch::Tensor t_indices,  // [n_rays, n_grids]
    const torch::Tensor hits,    // [n_rays, n_grids]
    // options
    const torch::Tensor near_planes,
    const torch::Tensor far_planes,
    const float step_size, // 默认是1e-3
    const float cone_angle,
    const bool compute_intervals,
    const bool compute_samples,
    const bool compute_terminate_planes,
    const int32_t traverse_steps_limit, // <= 0 means no limit
    const bool over_allocate) // over allocate the memory for intervals and samples
{
    DEVICE_GUARD(rays_o);
    if (over_allocate) {
        TORCH_CHECK(traverse_steps_limit > 0, "traverse_steps_limit must be > 0 when over_allocate is true");
    }

    int32_t n_rays = rays_o.size(0);
    int32_t n_grids = binaries.size(0);
    int3 resolution = make_int3(binaries.size(1), binaries.size(2), binaries.size(3));

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_threads = 512; 
    int32_t max_blocks = 65535;
    dim3 threads = dim3(min(max_threads, n_rays));
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(n_rays, threads.x)));

    // outputs
    RaySegmentsSpec intervals, samples;
    torch::Tensor terminate_planes;
    if (compute_terminate_planes) 
        terminate_planes = torch::empty({n_rays}, rays_o.options());

    if (over_allocate) {
        // over allocate the memory so that we can traverse the grids in a single pass.
        if (compute_intervals) {
            intervals.chunk_cnts = torch::full({n_rays}, traverse_steps_limit * 2, rays_o.options().dtype(torch::kLong)) * rays_mask;
            intervals.memalloc_data_from_chunk(true, true);
        }
        if (compute_samples) {
            samples.chunk_cnts = torch::full({n_rays}, traverse_steps_limit, rays_o.options().dtype(torch::kLong)) * rays_mask;
            samples.memalloc_data_from_chunk(false, true, true);
        }

        device::traverse_grids_kernel<<<blocks, threads, 0, stream>>>(
            // rays
            n_rays,
            rays_o.data_ptr<float>(),  // [n_rays, 3]
            rays_d.data_ptr<float>(),  // [n_rays, 3]
            rays_mask.data_ptr<bool>(),  // [n_rays]
            // grids
            n_grids,
            resolution,
            binaries.data_ptr<bool>(), // [n_grids, resx, resy, resz]
            aabbs.data_ptr<float>(),   // [n_grids, 6]
            // sorted intersections
            hits.data_ptr<bool>(),         // [n_rays, n_grids]
            t_sorted.data_ptr<float>(),    // [n_rays, n_grids * 2]
            t_indices.data_ptr<int64_t>(), // [n_rays, n_grids * 2]
            // options
            near_planes.data_ptr<float>(), // [n_rays]
            far_planes.data_ptr<float>(),  // [n_rays]
            step_size,
            cone_angle,
            traverse_steps_limit,
            // outputs
            false,
            device::PackedRaySegmentsSpec(intervals),
            device::PackedRaySegmentsSpec(samples),
            compute_terminate_planes ? terminate_planes.data_ptr<float>() : nullptr);
        
        // update the chunk starts with the actual chunk_cnts from traversal.
        intervals.compute_chunk_start();
        samples.compute_chunk_start();
    } else {
        // To allocate the accurate memory we need to traverse the grids twice.
        // The first pass is to count the number of segments along each ray.
        // The second pass is to fill the segments.
        if (compute_intervals)
            intervals.chunk_cnts = torch::empty({n_rays}, rays_o.options().dtype(torch::kLong));
        if (compute_samples)
            samples.chunk_cnts = torch::empty({n_rays}, rays_o.options().dtype(torch::kLong));
        device::traverse_grids_kernel<<<blocks, threads, 0, stream>>>(
            // rays
            n_rays,
            rays_o.data_ptr<float>(),  // [n_rays, 3]
            rays_d.data_ptr<float>(),  // [n_rays, 3]
            nullptr,  /* rays_mask */
            // grids
            n_grids,
            resolution,
            binaries.data_ptr<bool>(), // [n_grids, resx, resy, resz]
            aabbs.data_ptr<float>(),   // [n_grids, 6]
            // sorted intersections
            hits.data_ptr<bool>(),         // [n_rays, n_grids]
            t_sorted.data_ptr<float>(),    // [n_rays, n_grids * 2]
            t_indices.data_ptr<int64_t>(), // [n_rays, n_grids * 2]
            // options
            near_planes.data_ptr<float>(), // [n_rays]
            far_planes.data_ptr<float>(),  // [n_rays]
            step_size,
            cone_angle,
            traverse_steps_limit,
            // outputs
            true,
            device::PackedRaySegmentsSpec(intervals),
            device::PackedRaySegmentsSpec(samples),
            nullptr);  /* terminate_planes */
        
        // second pass to record the segments.
        if (compute_intervals)
            intervals.memalloc_data_from_chunk(true, true);
        if (compute_samples)
            samples.memalloc_data_from_chunk(false, false, true);
        device::traverse_grids_kernel<<<blocks, threads, 0, stream>>>(
            // rays
            n_rays,
            rays_o.data_ptr<float>(),  // [n_rays, 3]
            rays_d.data_ptr<float>(),  // [n_rays, 3]
            nullptr,  /* rays_mask */
            // grids
            n_grids,
            resolution,
            binaries.data_ptr<bool>(), // [n_grids, resx, resy, resz]
            aabbs.data_ptr<float>(),   // [n_grids, 6]
            // sorted intersections
            hits.data_ptr<bool>(),         // [n_rays, n_grids]
            t_sorted.data_ptr<float>(),    // [n_rays, n_grids * 2]
            t_indices.data_ptr<int64_t>(), // [n_rays, n_grids * 2]
            // options
            near_planes.data_ptr<float>(), // [n_rays]
            far_planes.data_ptr<float>(),  // [n_rays]
            step_size,
            cone_angle,
            traverse_steps_limit,
            // outputs
            false,
            device::PackedRaySegmentsSpec(intervals),
            device::PackedRaySegmentsSpec(samples),
            compute_terminate_planes ? terminate_planes.data_ptr<float>() : nullptr);
    }
    
    return {intervals, samples, terminate_planes};
}


std::vector<torch::Tensor> ray_aabb_intersect(
    const torch::Tensor rays_o, // [n_rays, 3]
    const torch::Tensor rays_d, // [n_rays, 3]
    const torch::Tensor aabbs,  // [n_aabbs, 6]
    const float near_plane,
    const float far_plane, 
    const float miss_value)  
{
    DEVICE_GUARD(rays_o);

    int32_t n_rays = rays_o.size(0);
    int32_t n_aabbs = aabbs.size(0);
    int32_t numel = n_rays * n_aabbs;

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_threads = 512; 
    int32_t max_blocks = 65535;
    dim3 threads = dim3(min(max_threads, numel));
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(numel, threads.x)));

    // outputs
    torch::Tensor t_mins = torch::empty({n_rays, n_aabbs}, rays_o.options());
    torch::Tensor t_maxs = torch::empty({n_rays, n_aabbs}, rays_o.options());
    torch::Tensor hits = torch::empty({n_rays, n_aabbs}, rays_d.options().dtype(torch::kBool));

    device::ray_aabb_intersect_kernel<<<blocks, threads, 0, stream>>>(
        // rays
        n_rays,
        rays_o.data_ptr<float>(),  // [n_rays, 3]
        rays_d.data_ptr<float>(),  // [n_rays, 3]
        near_plane,
        far_plane,
        // aabbs
        n_aabbs,
        aabbs.data_ptr<float>(),   // [n_aabbs, 6]
        // outputs
        miss_value,
        t_mins.data_ptr<float>(),   // [n_rays, n_aabbs]
        t_maxs.data_ptr<float>(),   // [n_rays, n_aabbs]
        hits.data_ptr<bool>());     // [n_rays, n_aabbs]

    return {t_mins, t_maxs, hits};
}

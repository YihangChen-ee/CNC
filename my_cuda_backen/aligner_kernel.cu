#include <torch/extension.h>


template <typename scalar_t, uint32_t num_dim>
__global__ void query_mask_3D_kernel_2D(
    const torch::PackedTensorAccessor<short, 2, torch::RestrictPtrTraits, size_t> points_n_orig,  // [N, 3]  x, y, z
    const torch::PackedTensorAccessor<bool, 2, torch::RestrictPtrTraits, size_t> binary_vxl,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> mask,
    torch::PackedTensorAccessor<int, 1, torch::RestrictPtrTraits, size_t> overlap_area_pool,
    const int resolution
){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int Rb = binary_vxl.size(0);
    const float Rb_re = 1.0 / float(Rb);

    if (i>=mask.size(0)) return;

    bool m = false;
    float scale_re = 1.0 / (float(resolution) - 2.0);
    uint32_t pos_g[num_dim*2];  // xmin, ymin, zmin, xmax, ymax, zmax
    float points_ns[num_dim] = {0};

    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {
        points_ns[d] = (float(points_n_orig[i][d]) - 0.5) * scale_re;
    }

    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {
        float pos_g1 = points_ns[d] - scale_re;
        pos_g1 = pos_g1 * Rb;
        pos_g1 = pos_g1 < 0? 0:pos_g1;
        pos_g1 = pos_g1 > Rb-1? Rb-1:pos_g1;
        pos_g[d] = int(pos_g1);

        float pos_g2 = points_ns[d] + scale_re;
        pos_g2 = pos_g2 * Rb;
        pos_g2 = pos_g2 < 0? 0:pos_g2;
        pos_g2 = pos_g2 > Rb-1? Rb-1:pos_g2;
        pos_g[num_dim + d] = int(pos_g2);
    }

    float right_a = 0;
    float left_a = 0;
    float overlap_a = 0;
    float right_b = 0;
    float left_b = 0;
    float overlap_b = 0;
    float right_c = 0;
    float left_c = 0;
    float overlap_c = 0;
    float overlap_area = 0;
    bool m_tmp = false;

    #pragma unroll
    for (int idx_a=pos_g[0]; idx_a<=pos_g[2]; idx_a++){
        right_a = min(float(idx_a)*Rb_re + Rb_re, points_ns[0] + scale_re);
        left_a = max(float(idx_a)*Rb_re, points_ns[0] - scale_re);
        overlap_a = (right_a - left_a);
        #pragma unroll
        for (int idx_b=pos_g[1]; idx_b<=pos_g[3]; idx_b++){
            right_b = min(float(idx_b)*Rb_re + Rb_re, points_ns[1] + scale_re);
            left_b = max(float(idx_b)*Rb_re, points_ns[1] - scale_re);
            overlap_b = (right_b - left_b);
            //
            m_tmp = (binary_vxl[idx_a][idx_b]);
            m = m | m_tmp;
            if (overlap_a < 0 || overlap_b < 0)
                printf("warning!!!!! overlap_area wrong!\n");
            if (m_tmp == true)
                overlap_area += overlap_a * overlap_b;
        }
    }

    overlap_area = overlap_area * Rb * Rb;

    mask[i] = int(m);
    // printf(" %f ", overlap_area);
    overlap_area_pool[i] = int(overlap_area * 1000);
}

template <typename scalar_t, uint32_t num_dim>
__global__ void query_mask_3D_kernel_2D_qlist(
    const torch::PackedTensorAccessor<short, 2, torch::RestrictPtrTraits, size_t> points_n_orig,  // [N, 3]  x, y, z
    const torch::PackedTensorAccessor<bool, 2, torch::RestrictPtrTraits, size_t> binary_vxl,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> mask,
    torch::PackedTensorAccessor<int, 1, torch::RestrictPtrTraits, size_t> overlap_area_pool,
    const torch::PackedTensorAccessor<long, 1, torch::RestrictPtrTraits, size_t> resolution_list  // [N]

){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int Rb = binary_vxl.size(0);
    const float Rb_re = 1.0 / float(Rb);

    if (i>=mask.size(0)) return;

    bool m = false;
    float scale_re = 1.0 / (float(resolution_list[i]) - 2.0);
    uint32_t pos_g[num_dim*2];  // xmin, ymin, zmin, xmax, ymax, zmax
    float points_ns[num_dim] = {0};

    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {
        points_ns[d] = (float(points_n_orig[i][d]) - 0.5) * scale_re;
    }

    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {
        float pos_g1 = points_ns[d] - scale_re;
        pos_g1 = pos_g1 * Rb;
        pos_g1 = pos_g1 < 0? 0:pos_g1;
        pos_g1 = pos_g1 > Rb-1? Rb-1:pos_g1;
        pos_g[d] = int(pos_g1);

        float pos_g2 = points_ns[d] + scale_re;
        pos_g2 = pos_g2 * Rb;
        pos_g2 = pos_g2 < 0? 0:pos_g2;
        pos_g2 = pos_g2 > Rb-1? Rb-1:pos_g2;
        pos_g[num_dim + d] = int(pos_g2);
    }

    float right_a = 0;
    float left_a = 0;
    float overlap_a = 0;
    float right_b = 0;
    float left_b = 0;
    float overlap_b = 0;
    float right_c = 0;
    float left_c = 0;
    float overlap_c = 0;
    float overlap_area = 0;
    bool m_tmp = false;

    #pragma unroll
    for (int idx_a=pos_g[0]; idx_a<=pos_g[2]; idx_a++){
        right_a = min(float(idx_a)*Rb_re + Rb_re, points_ns[0] + scale_re);
        left_a = max(float(idx_a)*Rb_re, points_ns[0] - scale_re);
        overlap_a = (right_a - left_a);
        #pragma unroll
        for (int idx_b=pos_g[1]; idx_b<=pos_g[3]; idx_b++){
            right_b = min(float(idx_b)*Rb_re + Rb_re, points_ns[1] + scale_re);
            left_b = max(float(idx_b)*Rb_re, points_ns[1] - scale_re);
            overlap_b = (right_b - left_b);
            //
            m_tmp = (binary_vxl[idx_a][idx_b]);
            m = m | m_tmp;
            if (overlap_a < 0 || overlap_b < 0)
                printf("warning!!!!! overlap_area wrong!\n");
            if (m_tmp == true)
                overlap_area += overlap_a * overlap_b;
        }
    }

    overlap_area = overlap_area * Rb * Rb;

    mask[i] = int(m);
    overlap_area_pool[i] = int(overlap_area * 1000);
}


template <typename scalar_t, uint32_t num_dim>
__global__ void query_mask_3D_kernel_3D(
    const torch::PackedTensorAccessor<short, 2, torch::RestrictPtrTraits, size_t> points_n_orig,  // [N, 3]  x, y, z
    const torch::PackedTensorAccessor<bool, 3, torch::RestrictPtrTraits, size_t> binary_vxl,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> mask,
    torch::PackedTensorAccessor<int, 1, torch::RestrictPtrTraits, size_t> overlap_area_pool,
    const int resolution
){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int Rb = binary_vxl.size(0);
    const float Rb_re = 1.0 / float(Rb);

    if (i>=mask.size(0)) return;

    bool m = false;
    float scale_re = 1.0 / (float(resolution) - 2.0);
    uint32_t pos_g[num_dim*2];  // xmin, ymin, zmin, xmax, ymax, zmax
    float points_ns[num_dim] = {0};

    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {
        points_ns[d] = (float(points_n_orig[i][d]) - 0.5) * scale_re;
    }

    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {
        float pos_g1 = points_ns[d] - scale_re;
        pos_g1 = pos_g1 * Rb;
        pos_g1 = pos_g1 < 0? 0:pos_g1;
        pos_g1 = pos_g1 > Rb-1? Rb-1:pos_g1;
        pos_g[d] = int(pos_g1);

        float pos_g2 = points_ns[d] + scale_re;
        pos_g2 = pos_g2 * Rb;
        pos_g2 = pos_g2 < 0? 0:pos_g2;
        pos_g2 = pos_g2 > Rb-1? Rb-1:pos_g2;
        pos_g[num_dim + d] = int(pos_g2);
    }

    float right_a = 0;
    float left_a = 0;
    float overlap_a = 0;
    float right_b = 0;
    float left_b = 0;
    float overlap_b = 0;
    float right_c = 0;
    float left_c = 0;
    float overlap_c = 0;
    float overlap_area = 0;
    bool m_tmp = false;

    #pragma unroll
    for (int idx_a=pos_g[0]; idx_a<=pos_g[3]; idx_a++){
        right_a = min(float(idx_a)*Rb_re + Rb_re, points_ns[0] + scale_re);
        left_a = max(float(idx_a)*Rb_re, points_ns[0] - scale_re);
        overlap_a = (right_a - left_a);
        #pragma unroll
        for (int idx_b=pos_g[1]; idx_b<=pos_g[4]; idx_b++){
            right_b = min(float(idx_b)*Rb_re + Rb_re, points_ns[1] + scale_re);
            left_b = max(float(idx_b)*Rb_re, points_ns[1] - scale_re);
            overlap_b = (right_b - left_b);
            #pragma unroll
            for (int idx_c=pos_g[2]; idx_c<=pos_g[5]; idx_c++){
                right_c = min(float(idx_c)*Rb_re + Rb_re, points_ns[2] + scale_re);
                left_c = max(float(idx_c)*Rb_re, points_ns[2] - scale_re);
                overlap_c = (right_c - left_c);
                //
                m_tmp = (binary_vxl[idx_a][idx_b][idx_c]);
                m = m | m_tmp;
                if (overlap_a < 0 || overlap_b < 0 || overlap_c < 0)
                    printf("warning!!!!! overlap_area wrong!\n");
                if (m_tmp == true)
                    overlap_area += overlap_a * overlap_b * overlap_c;
            }
        }
    }

    overlap_area = overlap_area * Rb * Rb * Rb;

    mask[i] = int(m);
    overlap_area_pool[i] = int(overlap_area * 1000);
}

template <typename scalar_t, uint32_t num_dim>
__global__ void query_mask_3D_kernel_3D_qlist(
    const torch::PackedTensorAccessor<short, 2, torch::RestrictPtrTraits, size_t> points_n_orig,  // [N, 3]  x, y, z
    const torch::PackedTensorAccessor<bool, 3, torch::RestrictPtrTraits, size_t> binary_vxl,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> mask,
    torch::PackedTensorAccessor<int, 1, torch::RestrictPtrTraits, size_t> overlap_area_pool,
    const torch::PackedTensorAccessor<long, 1, torch::RestrictPtrTraits, size_t> resolution_list
    // const int resolution
){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int Rb = binary_vxl.size(0);
    const float Rb_re = 1.0 / float(Rb);

    if (i>=mask.size(0)) return;

    bool m = false;
    float scale_re = 1.0 / (float(resolution_list[i]) - 2.0);
    uint32_t pos_g[num_dim*2];  // xmin, ymin, zmin, xmax, ymax, zmax
    float points_ns[num_dim] = {0};

    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {
        points_ns[d] = (float(points_n_orig[i][d]) - 0.5) * scale_re;
    }

    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {
        float pos_g1 = points_ns[d] - scale_re;
        pos_g1 = pos_g1 * Rb;
        pos_g1 = pos_g1 < 0? 0:pos_g1;
        pos_g1 = pos_g1 > Rb-1? Rb-1:pos_g1;
        pos_g[d] = int(pos_g1);

        float pos_g2 = points_ns[d] + scale_re;
        pos_g2 = pos_g2 * Rb;
        pos_g2 = pos_g2 < 0? 0:pos_g2;
        pos_g2 = pos_g2 > Rb-1? Rb-1:pos_g2;
        pos_g[num_dim + d] = int(pos_g2);
    }

    float right_a = 0;
    float left_a = 0;
    float overlap_a = 0;
    float right_b = 0;
    float left_b = 0;
    float overlap_b = 0;
    float right_c = 0;
    float left_c = 0;
    float overlap_c = 0;
    float overlap_area = 0;
    bool m_tmp = false;

    #pragma unroll
    for (int idx_a=pos_g[0]; idx_a<=pos_g[3]; idx_a++){
        right_a = min(float(idx_a)*Rb_re + Rb_re, points_ns[0] + scale_re);
        left_a = max(float(idx_a)*Rb_re, points_ns[0] - scale_re);
        overlap_a = (right_a - left_a);
        #pragma unroll
        for (int idx_b=pos_g[1]; idx_b<=pos_g[4]; idx_b++){
            right_b = min(float(idx_b)*Rb_re + Rb_re, points_ns[1] + scale_re);
            left_b = max(float(idx_b)*Rb_re, points_ns[1] - scale_re);
            overlap_b = (right_b - left_b);
            #pragma unroll
            for (int idx_c=pos_g[2]; idx_c<=pos_g[5]; idx_c++){
                right_c = min(float(idx_c)*Rb_re + Rb_re, points_ns[2] + scale_re);
                left_c = max(float(idx_c)*Rb_re, points_ns[2] - scale_re);
                overlap_c = (right_c - left_c);
                //
                m_tmp = (binary_vxl[idx_a][idx_b][idx_c]);
                m = m | m_tmp;
                if (overlap_a < 0 || overlap_b < 0 || overlap_c < 0)
                    printf("warning!!!!! overlap_area wrong!\n");
                if (m_tmp == true)
                    overlap_area += overlap_a * overlap_b * overlap_c;
            }
        }
    }

    overlap_area = overlap_area * Rb * Rb * Rb;

    mask[i] = int(m);
    overlap_area_pool[i] = int(overlap_area * 1000);
}

void query_mask_3D_cu(
    const torch::Tensor points_n_orig,  // [N, 3]  x, y, z
    const torch::Tensor binary_vxl,  // [128, 128, 128]
    torch::Tensor mask,
    torch::Tensor overlap_area_pool,
    const int resolution,
    const int N
){
    // torch::Tensor mask = torch::zeros({N}, points_n_orig.options());

    const dim3 threads(512);  // 256
    const dim3 blocks((N+threads.x-1)/threads.x, 1, 1);
    const uint32_t num_dim = points_n_orig.size(1);

    switch (num_dim) {
        case 2:
            AT_DISPATCH_INTEGRAL_TYPES(points_n_orig.scalar_type(), "query_mask_2D_cu",
            ([&] {
                    query_mask_3D_kernel_2D<scalar_t, 2><<<blocks, threads>>>(
                        points_n_orig.packed_accessor<short, 2, torch::RestrictPtrTraits, size_t>(),
                        binary_vxl.packed_accessor<bool, 2, torch::RestrictPtrTraits, size_t>(),
                        mask.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                        overlap_area_pool.packed_accessor<int, 1, torch::RestrictPtrTraits, size_t>(),
                        resolution
                    );
                })); break;
        case 3:
            AT_DISPATCH_INTEGRAL_TYPES(points_n_orig.scalar_type(), "query_mask_3D_cu",
            ([&] {
                    query_mask_3D_kernel_3D<scalar_t, 3><<<blocks, threads>>>(
                        points_n_orig.packed_accessor<short, 2, torch::RestrictPtrTraits, size_t>(),
                        binary_vxl.packed_accessor<bool, 3, torch::RestrictPtrTraits, size_t>(),
                        mask.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                        overlap_area_pool.packed_accessor<int, 1, torch::RestrictPtrTraits, size_t>(),
                        resolution
                    );
                })); break;
    }

}


void query_mask_3D_qlist_cu(
    const torch::Tensor points_n_orig_list,  // [N, 3]  x, y, z
    const torch::Tensor binary_vxl,  // [128, 128, 128]
    torch::Tensor mask,
    torch::Tensor overlap_area_pool,
    const torch::Tensor resolution_list,  // [N]
    const int N
){
    // torch::Tensor mask = torch::zeros({N}, points_n_orig_list.options());

    const dim3 threads(512);  // 256
    const dim3 blocks((N+threads.x-1)/threads.x, 1, 1);
    const uint32_t num_dim = points_n_orig_list.size(1);

    switch (num_dim) {
        case 2:
            AT_DISPATCH_INTEGRAL_TYPES(points_n_orig_list.scalar_type(), "query_mask_2D_qlist_cu",
            ([&] {
                    query_mask_3D_kernel_2D_qlist<scalar_t, 2><<<blocks, threads>>>(
                        points_n_orig_list.packed_accessor<short, 2, torch::RestrictPtrTraits, size_t>(),
                        binary_vxl.packed_accessor<bool, 2, torch::RestrictPtrTraits, size_t>(),
                        mask.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                        overlap_area_pool.packed_accessor<int, 1, torch::RestrictPtrTraits, size_t>(),
                        resolution_list.packed_accessor<long, 1, torch::RestrictPtrTraits, size_t>()
                    );
                })); break;
        case 3:
            AT_DISPATCH_INTEGRAL_TYPES(points_n_orig_list.scalar_type(), "query_mask_3D_qlist_cu",
            ([&] {
                    query_mask_3D_kernel_3D_qlist<scalar_t, 3><<<blocks, threads>>>(
                        points_n_orig_list.packed_accessor<short, 2, torch::RestrictPtrTraits, size_t>(),
                        binary_vxl.packed_accessor<bool, 3, torch::RestrictPtrTraits, size_t>(),
                        mask.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                        overlap_area_pool.packed_accessor<int, 1, torch::RestrictPtrTraits, size_t>(),
                        resolution_list.packed_accessor<long, 1, torch::RestrictPtrTraits, size_t>()
                    );
                })); break;
    }

}



template <typename scalar_t>
__global__ void align_and_pack_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> voxel_features,
    const torch::PackedTensorAccessor<long, 1, torch::RestrictPtrTraits, size_t> unique_count,
    const torch::PackedTensorAccessor<long, 1, torch::RestrictPtrTraits, size_t> unique_count_cumsum,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> packed_features,
    const float V
){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // ith hash param
    const int j = blockIdx.y * blockDim.y + threadIdx.y;  // param's jth voxel
    const int k = blockIdx.z * blockDim.z + threadIdx.z;  // kth feature_dim

    if (i>=packed_features.size(0) || j>=packed_features.size(1) || k>=packed_features.size(2)) return;


    if ((j + 1) > unique_count[i]){  // empty
        packed_features[i][j][k] = V;
    }
    else{
        packed_features[i][j][k] = voxel_features[unique_count_cumsum[i] + j][k];
    }

}

torch::Tensor align_and_pack_forward_cu(
    const torch::Tensor voxel_features,  // [unique_count.sum(), F]
    const torch::Tensor unique_count,  // [N]
    const torch::Tensor unique_count_cumsum,  // [N+1]
    const int N,
    const int M,
    const int F,
    const float V,
    const int dim
){
    torch::Tensor packed_features = torch::zeros({N, M, F}, voxel_features.options());

    if (dim == 3) {
        const dim3 threads(2, 128, 1);
        const dim3 blocks((N+threads.x-1)/threads.x, (M+threads.y-1)/threads.y, (F+threads.z-1)/threads.z);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(voxel_features.scalar_type(), "align_and_pack_forward_cu",
        ([&] {
            align_and_pack_forward_kernel<scalar_t><<<blocks, threads>>>(

                // voxel_features.data_ptr<scalar_t>(),
                // unique_count.data_ptr<int>(),
                // unique_count_cumsum.data_ptr<int>(),
                // packed_features.data_ptr<float>(),

                voxel_features.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                // unique_count.packed_accessor<int, 1, torch::RestrictPtrTraits>(),
                unique_count.packed_accessor<long, 1, torch::RestrictPtrTraits, size_t>(),
                unique_count_cumsum.packed_accessor<long, 1, torch::RestrictPtrTraits, size_t>(),
                packed_features.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                V
            );
        }));
    }
    else {
        const dim3 threads(64, 4, 1);
        const dim3 blocks((N+threads.x-1)/threads.x, (M+threads.y-1)/threads.y, (F+threads.z-1)/threads.z);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(voxel_features.scalar_type(), "align_and_pack_forward_cu",
        ([&] {
            align_and_pack_forward_kernel<scalar_t><<<blocks, threads>>>(

                // voxel_features.data_ptr<scalar_t>(),
                // unique_count.data_ptr<int>(),
                // unique_count_cumsum.data_ptr<int>(),
                // packed_features.data_ptr<float>(),

                voxel_features.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                // unique_count.packed_accessor<int, 1, torch::RestrictPtrTraits>(),
                unique_count.packed_accessor<long, 1, torch::RestrictPtrTraits, size_t>(),
                unique_count_cumsum.packed_accessor<long, 1, torch::RestrictPtrTraits, size_t>(),
                packed_features.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                V
            );
        }));
    }

    return packed_features;
}


template <typename scalar_t>
__global__ void align_and_pack_backward_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> dL_packed_features,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> voxel_features,
    const torch::PackedTensorAccessor<long, 1, torch::RestrictPtrTraits, size_t> unique_count,
    const torch::PackedTensorAccessor<long, 1, torch::RestrictPtrTraits, size_t> unique_count_cumsum,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_voxel_features
){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // ith hash param
    const int j = blockIdx.y * blockDim.y + threadIdx.y;  // param's jth voxel
    const int k = blockIdx.z * blockDim.z + threadIdx.z;  // kth feature_dim

    if (i>=dL_packed_features.size(0) || j>=dL_packed_features.size(1) || k>=dL_packed_features.size(2)) return;

    if ((j + 1) > unique_count[i]) return;  // invalid point

    dL_voxel_features[unique_count_cumsum[i]+j][k] = dL_packed_features[i][j][k];

}

torch::Tensor align_and_pack_backward_cu(
    const torch::Tensor dL_packed_features,  // [N, M, F]
    const torch::Tensor voxel_features,  // [unique_count.sum(), F]
    const torch::Tensor unique_count,  // [N]
    const torch::Tensor unique_count_cumsum,  // [N+1]
    const int N,
    const int M,
    const int F,
    const int T,
    const int dim
){
    torch::Tensor dL_voxel_features = torch::zeros({T, F}, dL_packed_features.options());

    if (dim == 3) {
        const dim3 threads(2, 128, 1);
        const dim3 blocks((N+threads.x-1)/threads.x, (M+threads.y-1)/threads.y, (F+threads.z-1)/threads.z);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(dL_packed_features.scalar_type(), "align_and_pack_backward_cu",
        ([&] {
            align_and_pack_backward_kernel<scalar_t><<<blocks, threads>>>(
                dL_packed_features.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                voxel_features.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                // unique_count.packed_accessor<int, 1, torch::RestrictPtrTraits>(),
                unique_count.packed_accessor<long, 1, torch::RestrictPtrTraits, size_t>(),
                unique_count_cumsum.packed_accessor<long, 1, torch::RestrictPtrTraits, size_t>(),
                dL_voxel_features.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
            );
        }));
    }
    else {
        const dim3 threads(64, 4, 1);
        const dim3 blocks((N+threads.x-1)/threads.x, (M+threads.y-1)/threads.y, (F+threads.z-1)/threads.z);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(dL_packed_features.scalar_type(), "align_and_pack_backward_cu",
        ([&] {
            align_and_pack_backward_kernel<scalar_t><<<blocks, threads>>>(
                dL_packed_features.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                voxel_features.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                // unique_count.packed_accessor<int, 1, torch::RestrictPtrTraits>(),
                unique_count.packed_accessor<long, 1, torch::RestrictPtrTraits, size_t>(),
                unique_count_cumsum.packed_accessor<long, 1, torch::RestrictPtrTraits, size_t>(),
                dL_voxel_features.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
            );
        }));
    }

    return dL_voxel_features;
}
#include <torch/extension.h>
#include "utils.h"

torch::Tensor align_and_pack_forward(
    const torch::Tensor voxel_features,
    const torch::Tensor unique_count,
    const torch::Tensor unique_count_cumsum,
    const int N,
    const int M,
    const int F,
    const float V,
    const int dim
){
    CHECK_INPUT(voxel_features);
    CHECK_INPUT(unique_count);
    return align_and_pack_forward_cu(voxel_features, unique_count, unique_count_cumsum, N, M, F, V, dim);
}

torch::Tensor align_and_pack_backward(
    const torch::Tensor dL_packed_features,
    const torch::Tensor voxel_features,
    const torch::Tensor unique_count,
    const torch::Tensor unique_count_cumsum,
    const int N,
    const int M,
    const int F,
    const int T,
    const int dim
){
    CHECK_INPUT(dL_packed_features);
    CHECK_INPUT(voxel_features);
    CHECK_INPUT(unique_count);
    CHECK_INPUT(unique_count_cumsum);
    return align_and_pack_backward_cu(dL_packed_features, voxel_features, unique_count, unique_count_cumsum, N, M, F, T, dim);
}

void query_mask_3D(
    const torch::Tensor points_n_orig,
    const torch::Tensor binary_vxl,
    torch::Tensor mask,
    torch::Tensor overlap_area_pool,
    const int resolution,
    const int N
){
    CHECK_INPUT(points_n_orig);
    CHECK_INPUT(binary_vxl);
    CHECK_INPUT(mask);
    CHECK_INPUT(overlap_area_pool);

    query_mask_3D_cu(points_n_orig, binary_vxl, mask, overlap_area_pool, resolution, N);

}


void query_mask_3D_qlist(
    const torch::Tensor points_n_orig_list,
    const torch::Tensor binary_vxl,
    torch::Tensor mask,
    torch::Tensor overlap_area_pool,
    const torch::Tensor resolution_list,
    const int N
){
    CHECK_INPUT(points_n_orig_list);
    CHECK_INPUT(binary_vxl);
    CHECK_INPUT(mask);
    CHECK_INPUT(overlap_area_pool);
    CHECK_INPUT(resolution_list);

    query_mask_3D_qlist_cu(points_n_orig_list, binary_vxl, mask, overlap_area_pool, resolution_list, N);

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("align_and_pack_forward", &align_and_pack_forward);
    m.def("align_and_pack_backward", &align_and_pack_backward);
    m.def("query_mask_3D", &query_mask_3D);
    m.def("query_mask_3D_qlist", &query_mask_3D_qlist);
}
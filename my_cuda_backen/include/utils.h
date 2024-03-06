#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor align_and_pack_forward_cu(
    const torch::Tensor voxel_features,
    const torch::Tensor unique_count,
    const torch::Tensor unique_count_cumsum,
    const int N,
    const int M,
    const int F,
    const float V,
    const int dim
);


torch::Tensor align_and_pack_backward_cu(
    const torch::Tensor dL_packed_features,
    const torch::Tensor voxel_features,
    const torch::Tensor unique_count,
    const torch::Tensor unique_count_cumsum,
    const int N,
    const int M,
    const int F,
    const int T,
    const int dim
);


void query_mask_3D_cu(
    const torch::Tensor points_n_orig,
    const torch::Tensor binary_vxl,
    torch::Tensor mask,
    torch::Tensor overlap_area_pool,
    const int resolution,
    const int N
);

void query_mask_3D_qlist_cu(
    const torch::Tensor points_n_orig_list,
    const torch::Tensor binary_vxl,
    torch::Tensor mask,
    torch::Tensor overlap_area_pool,
    const torch::Tensor resolution_list,
    const int N
);
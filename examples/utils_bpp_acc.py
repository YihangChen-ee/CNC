import os
import random
import time
import torchac
import torch
from torch import Tensor
import pack_and_align
import _gridencoder as _backend
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import torch.nn.functional as nnf
import numpy as np
from utils import (
    get_grid_index,
)

MINS_LOG2_REP = -1/np.log(2)
TORCH_0 = torch.tensor([0], device='cuda')
TORCH_N_LIST = [torch.tensor([n], device='cuda') for n in range(0, 100)]

def get_time():
    torch.cuda.synchronize()
    p = time.perf_counter()
    return p

class _cnt_np_embed(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, embeddings, resolution, hashmap_size, axis):
        axis_list = ['xy', 'xz', 'yz']
        N = inputs.shape[0]
        n_features = embeddings.shape[-1]
        assert inputs.shape[-1] == 3
        inputs = inputs.to(torch.int16)
        assert axis in axis_list
        axis = axis_list.index(axis)
        scale = resolution - 2
        pn_embed = torch.zeros(size=[scale, scale, n_features, 2], device=inputs.device)
        _backend.cnt_np_embed(
            inputs,
            embeddings,
            pn_embed,
            N, resolution, n_features, hashmap_size, axis
        )
        pn_embed_sum = torch.sum(pn_embed, dim=-1, keepdim=True) + 1e-6  # [512, 512, 4, 1]

        # for pn_embed_frac[..., 0]: = (P1+P2+...)/pn_embed_sum[..., 0]; Pn=1
        # for pn_embed_frac[..., 1]: = -(N1+N2+...)/pn_embed_sum[..., 0]; Nn=-1
        # -> d pn_embed_frac[..., 0] / d Pn = 1/pn_embed_sum[..., 0]
        # -> d pn_embed_frac[..., 1] / d Nn = -1/pn_embed_sum[..., 0]
        pn_embed_frac = pn_embed / pn_embed_sum  # [512, 512, 4, 2]
        ctx.save_for_backward(inputs, embeddings, pn_embed_sum)
        ctx.dims = [N, resolution, n_features, hashmap_size, axis]
        return pn_embed_frac

    @staticmethod
    # @once_differentiable
    @custom_bwd
    def backward(ctx, grad):
        # grad: [512, 512, 4, 2]
        inputs, embeddings, pn_embed_sum = ctx.saved_tensors
        N, resolution, n_features, hashmap_size, axis = ctx.dims
        grad_embeddings = torch.zeros_like(embeddings)

        _backend.cnt_np_embed_backward(
            inputs,
            embeddings,
            pn_embed_sum,
            grad,
            grad_embeddings,
            N, resolution, n_features, hashmap_size, axis
        )

        return None, grad_embeddings, None, None, None

def encoder(x, p, file_name):
    x = x.detach().cpu()
    p = p.detach().cpu()
    assert file_name[-2:] == '.b'
    p_u = 1 - p.unsqueeze(-1)
    p_0 = torch.zeros_like(p_u)
    p_1 = torch.ones_like(p_u)
    # Encode to bytestream.
    output_cdf = torch.cat([p_0, p_u, p_1], dim=-1)  # Get CDF from your model, shape B, C, H, W, Lp
    sym = ((x+1)//2).to(torch.int16)  # Get the symbols to encode, shape B, C, H, W.
    byte_stream = torchac.encode_float_cdf(output_cdf, sym, check_input_bounds=True)
    # Number of bits taken by the stream
    bit_len = len(byte_stream) * 8
    # Write to a file.
    with open(file_name, 'wb') as fout:
        fout.write(byte_stream)
    return bit_len

def decoder(p, file_name):
    dvc = p.device
    p = p.detach().cpu()
    assert file_name[-2:] == '.b'
    p_u = 1 - p.unsqueeze(-1)
    p_0 = torch.zeros_like(p_u)
    p_1 = torch.ones_like(p_u)
    # Encode to bytestream.
    output_cdf = torch.cat([p_0, p_u, p_1], dim=-1)  # Get CDF from your model, shape B, C, H, W, Lp
    # Read from a file.
    with open(file_name, 'rb') as fin:
        byte_stream = fin.read()
    # Decode from bytestream.
    sym_out = torchac.decode_float_cdf(output_cdf, byte_stream)
    sym_out = (sym_out * 2 - 1).to(torch.float32)
    return sym_out.to(dvc)


class align_and_pack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voxel_features, unique_cnt, V, dim=3):

        voxel_features = voxel_features.contiguous()
        unique_count_cumsum = torch.cumsum(unique_cnt, dim=0)
        unique_count_cumsum = torch.cat([TORCH_0, unique_count_cumsum])
        N = unique_cnt.numel()
        M = unique_cnt.max()
        F = voxel_features.shape[-1]
        T = unique_count_cumsum[-1]

        packed_features = pack_and_align.align_and_pack_forward(voxel_features, unique_cnt, unique_count_cumsum, N, M, F, 0.0, dim)

        ctx.save_for_backward(voxel_features, unique_cnt, unique_count_cumsum)
        ctx.dims = [N, M, F, T, dim]

        return packed_features
    @staticmethod
    def backward(ctx, dL_packed_features):
        dL_packed_features = dL_packed_features.contiguous()
        voxel_features, unique_cnt, unique_count_cumsum = ctx.saved_tensors
        N, M, F, T, dim = ctx.dims
        # dL_packed_features = dL_packed_features / unique_cnt.unsqueeze(-1).unsqueeze(-1) * M
        dL_voxel_features = pack_and_align.align_and_pack_backward(dL_packed_features, voxel_features, unique_cnt,
                                                        unique_count_cumsum, N, M, F, T, dim)
        return dL_voxel_features, None, None, None


def my_meshgrid3D(start = (0, 0, 0), end = (1000, 1000, 1000), dtype=torch.int32):
    if isinstance(start, int):
        sx = sy = sz = start
        ex = ey = ez = end
        lx = ly = lz = end - start
    else:
        sx, sy, sz = start
        ex, ey, ez = end
        lx, ly, lz = ex - sx, ey - sy, ez - sz
    x = torch.arange(sx, ex, device='cuda', dtype=dtype)
    y = torch.arange(sy, ey, device='cuda', dtype=dtype)
    z = torch.arange(sz, ez, device='cuda', dtype=dtype)

    xx = x.unsqueeze(-1).unsqueeze(-1).repeat(1, ly, lz)
    yy = y.unsqueeze(0).unsqueeze(-1).repeat(lx, 1, lz)
    zz = z.unsqueeze(0).unsqueeze(0).repeat(lx, ly, 1)

    points = torch.stack([xx, yy, zz], dim=-1)  # [r, r, r, 3]

    return points


class STE_binary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = torch.clamp(input, min=-1, max=1)
        # out = torch.sign(input)
        p = (input >= 0) * (+1.0)
        n = (input < 0) * (-1.0)
        out = p + n
        return out
    @staticmethod
    def backward(ctx, grad_output):
        # mask: to ensure x belongs to (-1, 1)
        input, = ctx.saved_tensors
        i2 = input.clone().detach()
        i3 = torch.clamp(i2, -1, 1)
        mask = (i3 == i2) + 0.0
        return grad_output * mask


class STE_multistep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, Q):
        return torch.round(input*Q)/Q
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class CNC_context_models(nn.Module):
    def __init__(self,
                 num_dim=3,
                 resolutions_list=(16, 22, 31, 42, 57, 78, 106, 146, 199, 273, 374, 512),
                 resolutions_list_2D=(128, 256, 512, 1024),
                 log2_hashmap_size=19,
                 log2_hashmap_size_2D=21,
                 n_features=4,
                 sample_num = 20000,
                 max_context_layer_num = 3,
                 ste_binary = False,
                 ste_multistep = False,
                 add_noise = False,
                 Q = 100,
                 quantize_epoch = 1000,
                 Pg_level = -1,
                 Pg_level_2D = -1,
                 Rb = 128,
                 step_update = 16,
                 skip_levels_3D = (0, 1, 2, 3),
                 skip_levels_2D = (0, ),
                 use_dimension_wise=True,
                 use_overlap_area_pool=True,
                 ):
        super().__init__()

        self.MAX_POINTS_NUM_TO_OOM = 20000000
        self.use_overlap_area_pool = use_overlap_area_pool
        self.use_dimension_wise = use_dimension_wise

        resolutions_list = torch.tensor(resolutions_list).cuda()
        n_levels = resolutions_list.numel()

        resolutions_list_2D = torch.tensor(resolutions_list_2D).cuda()
        n_levels_2D = resolutions_list_2D.numel()

        self.num_dim = num_dim
        self.resolutions_list = resolutions_list
        self.resolutions_list_2D = resolutions_list_2D
        self.scales_list = (resolutions_list - 2).unsqueeze(-1)
        self.scales_list_2D = (resolutions_list_2D - 2).unsqueeze(-1)
        self.log2_hashmap_size = log2_hashmap_size
        self.log2_hashmap_size_2D = log2_hashmap_size_2D
        self.n_features = n_features
        self.n_levels = n_levels
        self.n_levels_2D = n_levels_2D
        self.ste_binary = ste_binary
        self.ste_multistep = ste_multistep
        self.add_noise = add_noise
        self.Q = Q
        self.quantize_epoch = quantize_epoch
        if Pg_level == -1 or Pg_level >= n_levels:
            Pg_level = n_levels + 0
        if Pg_level <= 1:
            Pg_level = 1
        self.Pg_level = Pg_level
        self.skip_levels_3D = skip_levels_3D
        if Pg_level_2D == -1 or Pg_level_2D >= n_levels_2D:
            Pg_level_2D = n_levels_2D + 0
        if Pg_level_2D <= 1:
            Pg_level_2D = 1
        self.Pg_level_2D = Pg_level_2D
        self.skip_levels_2D = skip_levels_2D
        self.quantize_epoch_cnt = 0
        self.sample_num = sample_num
        self.max_context_layer_num = max_context_layer_num

        # 3D
        offsets_list = []
        offset = 0
        max_params = 2 ** log2_hashmap_size
        for i in range(n_levels):
            resolution = resolutions_list[i].item()
            params_in_level = min(max_params, resolution ** num_dim)  # limit max number
            params_in_level = int(np.ceil(params_in_level / 8) * 8)  # make divisible
            offsets_list.append(offset)
            offset += params_in_level
        offsets_list.append(offset)
        offsets_list = torch.from_numpy(np.array(offsets_list, dtype=np.int32)).cuda().to(torch.long)
        self.offsets_list = offsets_list

        # 2D
        offsets_list_2D = []
        offset_2D = 0
        max_params_2D = 2 ** log2_hashmap_size_2D
        for i in range(n_levels_2D):
            resolution_2D = resolutions_list_2D[i].item()
            params_in_level_2D = min(max_params_2D, resolution_2D ** 2)  # limit max number
            params_in_level_2D = int(np.ceil(params_in_level_2D / 8) * 8)  # make divisible
            offsets_list_2D.append(offset_2D)
            offset_2D += params_in_level_2D
        offsets_list_2D.append(offset_2D)
        offsets_list_2D = torch.from_numpy(np.array(offsets_list_2D, dtype=np.int32)).cuda().to(torch.long)
        self.offsets_list_2D = offsets_list_2D

        self.n_levels_thresh = self.n_levels - 1
        self.resolution_thresh = self.resolutions_list[-1]
        for i in range(n_levels - 1):
            if resolutions_list[i] ** num_dim <= max_params and resolutions_list[i+1] ** num_dim > max_params:
                self.n_levels_thresh = i + 1
                self.resolution_thresh = resolutions_list[i] + 0.0
        # ---

        self.unique_value_list = []
        self.unique_count_cumsum_list = []
        self.unique_count_list = []
        self.pos_grid_sorted_list = []

        for i in reversed(range(Pg_level)):

            current_resolution = resolutions_list[i].item()
            pos_grid = my_meshgrid3D(0, current_resolution).view(-1, 3).unsqueeze(1)  # [reso*reso*reso, 1, 3]
            indexes = get_grid_index(offsets_list[i+1] - offsets_list[i], current_resolution, pos_grid)[:, 0]  # [512*512*512]
            indexes_sotred, indices = torch.sort(indexes, descending=False, dim=0)
            pos_grid = pos_grid.to(torch.int16)
            pos_grid_sorted = torch.index_select(pos_grid[:, 0, :], dim=0, index=indices)  # [[reso*reso*reso, 3]
            unique_value, unique_cnt = torch.unique(indexes_sotred, return_counts=True)

            if current_resolution <= self.resolution_thresh:
                shuffle_idx = torch.randperm(unique_value.nelement())
                unique_value = unique_value[shuffle_idx]
                pos_grid_sorted = pos_grid_sorted[shuffle_idx]
                unique_cnt = unique_cnt[shuffle_idx]

            unique_count_cumsum = torch.cumsum(unique_cnt, dim=0)  # unique_count_cumsum: [520000]
            unique_count_cumsum = torch.cat([TORCH_0, unique_count_cumsum])
            unique_value = unique_value.to(torch.long)
            unique_count_cumsum = unique_count_cumsum.to(torch.long)

            self.unique_value_list.insert(0, unique_value)
            self.unique_count_cumsum_list.insert(0, unique_count_cumsum)
            self.unique_count_list.insert(0, unique_cnt)
            self.pos_grid_sorted_list.insert(0, pos_grid_sorted)

            del pos_grid
            del indexes
            del indexes_sotred
            del indices
            del pos_grid_sorted
            del unique_value
            del unique_cnt
            del unique_count_cumsum
            torch.cuda.empty_cache()

        tmp_lens = [self.unique_count_cumsum_list[i].numel() for i in range(Pg_level)]
        unique_count_cumsum_list = torch.zeros(size=[Pg_level,  max(tmp_lens)], device='cuda').to(torch.long)
        unique_count_list = torch.zeros(size=[Pg_level,  max(tmp_lens)], device='cuda').to(torch.long)
        for i in range(Pg_level):
            unique_count_cumsum_list[i, :tmp_lens[i]] = self.unique_count_cumsum_list[i]
            unique_count_list[i, :tmp_lens[i] - 1] = self.unique_count_list[i]
        self.unique_count_cumsum_list = unique_count_cumsum_list
        self.unique_count_list = unique_count_list

        del unique_count_cumsum_list
        del unique_count_list
        torch.cuda.empty_cache()
        # ---
        hashparams_num_levels = torch.tensor([tmp_lens[i]-1 for i in range(Pg_level)]).cuda()
        sample_num_levels = torch.round((hashparams_num_levels * (sample_num / hashparams_num_levels.sum()))).to(torch.long)
        sample_num_levels = hashparams_num_levels if sample_num_levels[-1] > hashparams_num_levels[-1] else sample_num_levels

        self.hashparams_num_levels = hashparams_num_levels
        self.ttl_hashparams_num_levels = torch.sum(hashparams_num_levels).item()
        self.ttl_hashparams_num_valid_levels = 0
        for n in range(self.n_levels):
            if n not in self.skip_levels_3D and n < self.Pg_level:
                self.ttl_hashparams_num_valid_levels += self.hashparams_num_levels[n].item()

        self.sample_num_levels = sample_num_levels
        self.ttl_sample_num = torch.sum(sample_num_levels).item()
        self.ttl_sample_num_valid_levels = 0
        for n in range(self.n_levels):
            if n not in self.skip_levels_3D and n < self.Pg_level:
                self.ttl_sample_num_valid_levels += self.sample_num_levels[n].item()

        self.utils_rand = torch.rand(size=[Pg_level]).cuda()
        self.utils_nlevel_idx = torch.tensor(range(Pg_level)).cuda()
        self.utils_points_per_param_levels = [((self.resolutions_list[i]**num_dim)/self.hashparams_num_levels[i]).item() for i in range(Pg_level)]

        x = torch.arange(0, Rb, device='cuda', dtype=torch.int32)
        y = torch.arange(0, Rb, device='cuda', dtype=torch.int32)
        xx = x.unsqueeze(-1).repeat(1, Rb)
        yy = y.unsqueeze(0).repeat(Rb, 1)
        self.binary_vxl_2D_idx = torch.stack([xx, yy], dim=-1)  # [r, r, 2]

        self.context_model_3D = nn.Sequential(
            nn.Linear(n_features * max_context_layer_num + 1, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, n_features),
        ).cuda()

        self.context_model_2D = []
        for n in range(1, Pg_level_2D):
            context_layer_num = min(n, max_context_layer_num)
            tmp = nn.Sequential(
                nn.Linear(n_features * (context_layer_num + int(use_dimension_wise)) + 1, n_features),
            )
            self.context_model_2D.append(tmp)
        self.context_model_2D = nn.Sequential(*self.context_model_2D).cuda()

        self.entropy_model = Bernoulli_entropy().cuda()

        self.binary_vxl_len = 128
        self.init_binary_vxl_coords()

        self.step_update = step_update
        self.idx_coords2_tmp = None
        self.batched_inputs_list = None

    def query_binary_vxl(self, points_n_orig, binary_vxl, n, mem_save=False, verbose=False, return_overlap_area=False):

        Rresolution = self.resolutions_list[n]
        N = points_n_orig.shape[0]
        mask = torch.zeros(size=[N], dtype=torch.int16, device=points_n_orig.device)
        overlap_area_pool = torch.zeros(size=[N], dtype=torch.int32, device=points_n_orig.device)
        pack_and_align.query_mask_3D(points_n_orig, binary_vxl.squeeze(0), mask, overlap_area_pool, Rresolution, N)
        mask = mask.to(torch.bool)

        if return_overlap_area:
            return mask, overlap_area_pool
        return mask

    def query_binary_vxl_qlist(self, points_n_orig_list, binary_vxl, n_list, return_overlap_area=False):

        Rresolution_list = self.resolutions_list[n_list]
        N = points_n_orig_list.shape[0]
        mask = torch.zeros(size=[N], dtype=torch.int16, device=points_n_orig_list.device)
        overlap_area_pool = torch.zeros(size=[N], dtype=torch.int32, device=points_n_orig_list.device)
        pack_and_align.query_mask_3D_qlist(points_n_orig_list, binary_vxl.squeeze(0), mask, overlap_area_pool, Rresolution_list, N)
        mask = mask.to(torch.bool)

        if return_overlap_area:
            return mask, overlap_area_pool
        return mask


    def fetch_2D_batches(self, binary_vxl_2D, n):

        Rb = binary_vxl_2D.shape[-1]
        T = self.scales_list_2D[n] / Rb
        assert T%1 == 0
        T = int(T)
        Rresolution = self.resolutions_list_2D[n]
        binary_vxl_2D = binary_vxl_2D.view(-1)  # [Rb*Rb]

        binary_vxl_2D_idx = self.binary_vxl_2D_idx.view(-1, 2)  # [r*r, 2]
        binary_vxl_2D_idx_1 = binary_vxl_2D_idx[binary_vxl_2D==1]  # [bs, 2]
        binary_vxl_2D_idx_1 = binary_vxl_2D_idx_1.unsqueeze(1).unsqueeze(1)  # [bs, 1, 1, 2]
        binary_vxl_2D_idx_1 = binary_vxl_2D_idx_1*T  # [bs, 1, 1, 2]

        RESOLUTION_OFFSETS = torch.tensor([[i, j] for i in range(-1, -1+T+2) for j in range(-1, -1+T+2)], device='cuda') + 1  # [(T+2)**2, 2]
        RESOLUTION_OFFSETS = RESOLUTION_OFFSETS.view(1, T+2, T+2, 2)  # [1, T+2, T+2, 2]

        points_n_orig = binary_vxl_2D_idx_1 + RESOLUTION_OFFSETS + 0.0   # [bs, T+2, T+2, 2]
        #
        points_n_orig = points_n_orig.to(torch.long)
        pos_grid_2D = points_n_orig.view(-1, 2).unsqueeze(1)  # [bs*(T+2)*(T+2), 2]
        indexes_2D = get_grid_index(self.offsets_list_2D[n+1] - self.offsets_list_2D[n], Rresolution.item(), pos_grid_2D)[:, 0]  # [bs*(T+2)*(T+2)]

        points_n = (points_n_orig - 0.5) / self.scales_list_2D[n].item()   # [bs, T+2, T+2, 2]

        return indexes_2D, points_n.view(-1, 2)


    def get_STE_params(self, Encoding, mode='ste_binary'):
        assert mode in ['ste_binary', 'ste_multistep', 'add_noise']
        params = Encoding.params
        params_q = params
        if mode == 'ste_binary':
            params_q = STE_binary.apply(params_q)
        elif mode == 'ste_multistep':
            params_q = STE_multistep.apply(params_q, self.Q)
        elif mode == 'add_noise':
            params_q = params_q + (torch.rand_like(params_q) - 0.5) * (1/self.Q)
        return params_q


    def get_BiRF_wentropy_leveln(self, params_q, n, offsets_list=None):
        if offsets_list is None:
            offsets_list = self.offsets_list
        params_q_n = params_q[offsets_list[n]:offsets_list[n + 1]]
        ttl_num_n = params_q_n.numel()
        tmp_n = torch.sum(params_q_n)
        pos_num_n = (ttl_num_n + tmp_n) / 2.0
        neg_num_n = (ttl_num_n - tmp_n) / 2.0
        Pg_n = pos_num_n / ttl_num_n
        pos_prob_n = Pg_n
        neg_prob_n = (1 - Pg_n)
        pos_bit_n = pos_num_n * (-torch.log2(pos_prob_n))
        neg_bit_n = neg_num_n * (-torch.log2(neg_prob_n))
        ttl_bit_n = pos_bit_n + neg_bit_n
        return Pg_n, ttl_bit_n, ttl_num_n


    def init_binary_vxl_coords(self, scale=512):
        t = scale // self.binary_vxl_len
        resolution = scale + 2
        self.idx_coord_base = my_meshgrid3D(-1, t + 1).unsqueeze(0)  # [1, 6, 6, 6, 3]
        self.idx_coord_temp = my_meshgrid3D(0, self.binary_vxl_len).view(-1, 3).unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [128*128*128, 1, 1, 1, 3]
        self.pn_frac_offsets_list = torch.tensor([0, resolution*resolution], device='cuda', dtype=torch.int32)
        self.pn_frac_resolutions_list = torch.tensor([resolution], device='cuda', dtype=torch.int32)


    def get_idx_coords2(self, binary_vxl, resolution=514):
        t = (resolution - 2) // self.binary_vxl_len
        binary_vxl_view = binary_vxl.squeeze(0).view(-1)  # [128*128*128]
        idx_coord_select = self.idx_coord_temp[binary_vxl_view]  # [N, 1, 1, 1, 3]
        idx_coords_orig = idx_coord_select * t + self.idx_coord_base  # [N, 6, 6, 6, 3]
        idx_coords_orig = idx_coords_orig.view(-1, 3) + 1  #   # [N*6*6*6, 3]. value: [0, 1, 2, 3, ..., 513]

        idx_coords_orig_tmp = idx_coords_orig[..., 0] * resolution * resolution + idx_coords_orig[..., 1] * resolution + idx_coords_orig[..., 2]  # [N*6*6*6]
        idx_coords2 = (torch.unique(idx_coords_orig_tmp, dim=0))
        idx_coords2_x = idx_coords2 // (resolution * resolution)
        idx_coords2_y = (idx_coords2 // resolution) % resolution
        idx_coords2_z = idx_coords2 % resolution
        idx_coords2 = torch.stack([idx_coords2_x, idx_coords2_y, idx_coords2_z], dim=-1)

        return idx_coords2


    def get_pn_embed_frac(self, embeddings_3D_q, idx_coords2, resolution=514, axis='xy'):
        hashmap_size = 2**self.log2_hashmap_size

        pn_embed_frac = _cnt_np_embed.apply(
            idx_coords2,
            embeddings_3D_q,
            resolution, hashmap_size, axis
        )  # [512, 512, 4, 2]

        pn_embed_frac = pn_embed_frac[..., 0]  # [512, 512, 4]
        pn_embed_frac = pn_embed_frac.unsqueeze(0).permute(0, 3, 1, 2).contiguous()  # [1, 4, 512, 512]
        pn_embed_frac = nnf.pad(pn_embed_frac, pad=[1, 1, 1, 1])  # [1, 4, 514, 514]
        pn_embed_frac = pn_embed_frac.squeeze(0).permute(1, 2, 0).contiguous()  # [514, 514, 4]
        pn_embed_frac = pn_embed_frac.view(-1, self.n_features)  # [514*514, 4]

        return pn_embed_frac


    def forward_binary_vxl_mixPg_3D2D(self, Encoding_xyz, Encoding_xy, Encoding_xz, Encoding_yz, binary_vxl=None, verbose=False, sample_num=None, step=0):

        def forward_2D(params_q_3D_clip, idx_coords2, Encoding_2D, params_q_2D, binary_vxl_2D, axis, batch_info):
            if self.use_dimension_wise:
                pn_embed_frac = self.get_pn_embed_frac(params_q_3D_clip, idx_coords2, axis=axis)  # [514*514, 4]
            ttl_bit = 0
            bi_idx = 0
            for n in range(self.n_levels_2D):
                Pg_n, ttl_bit_n, ttl_num_n = self.get_BiRF_wentropy_leveln(params_q_2D, n, self.offsets_list_2D)
                if n in self.skip_levels_2D or n >= self.Pg_level_2D:
                    pass
                else:

                    points_n, indices_2D, unique_value_2D, unique_cnt_2D = batch_info[bi_idx]
                    bi_idx += 1

                    context_layer_num = min(n, self.max_context_layer_num)

                    context = Encoding_2D(points_n, n-context_layer_num, n, binary_vxl=binary_vxl_2D, PV=0)  # [bs*(T+2)*(T+2), n_features*context_layer_num]
                    Pg_n = Pg_n.unsqueeze(-1).unsqueeze(-1).repeat(context.shape[0], 1)  # [N, 1]
                    if self.use_dimension_wise:
                        context_pn = Encoding_2D.forward_given_params(points_n,
                                                                      self.pn_frac_offsets_list,
                                                                      self.pn_frac_resolutions_list,
                                                                      pn_embed_frac, binary_vxl_2D)#.detach()  # [bs*(T+2)*(T+2), n_features]
                        context = torch.cat([context, context_pn, Pg_n], dim=-1)
                    else:
                        context = torch.cat([context, Pg_n], dim=-1)
                    mean = self.context_model_2D[n - 1](context)  # [bs*(T+2)*(T+2), n_features*2]

                    mean = torch.index_select(mean, dim=0, index=indices_2D)  # [bs*(T+2)*(T+2), n_features]
                    mean = align_and_pack.apply(mean, unique_cnt_2D, 0.0, 2)  # [num_valid, unique_cnt_2D.max(), n_features]
                    mean = torch.sum(mean, dim=1)  # [num_valid, n_features]
                    mean = mean / unique_cnt_2D.unsqueeze(-1)  # [num_valid, n_features]
                    values_q = params_q_2D[unique_value_2D, :]  # [unique(bs*(T+2)*(T+2)), n_features]

                    params_bit_n = self.entropy_model.forward(values_q, mean)
                    ttl_bit_n = torch.sum(params_bit_n)
                ttl_bit += ttl_bit_n

            ttl_num = params_q_2D.numel()
            return ttl_bit, ttl_num

        # 2D
        params_q_xy = self.get_STE_params(Encoding_xy)
        params_q_xz = self.get_STE_params(Encoding_xz)
        params_q_yz = self.get_STE_params(Encoding_yz)
        # 3D
        params_q_xyz = self.get_STE_params(Encoding_xyz)

        ttl_bit_sum = 0
        ttl_num_sum = 0

        if step % self.step_update == 0:
            idx_coords2 = self.get_idx_coords2(binary_vxl)
            self.idx_coords2_tmp = idx_coords2
        else:
            idx_coords2 = self.idx_coords2_tmp

        binary_vxl_2D_list = []
        for axis_dim in (2, 1, 0):  # 'xy', 'xz', 'yz'
            binary_vxl_2D_list.append(torch.any(binary_vxl.squeeze(0), dim=axis_dim))

        if step % self.step_update == 0:
            batched_inputs_list = [[], [], []]
            for axis_dim in (0, 1, 2):
                for n in range(self.n_levels_2D):
                    if n in self.skip_levels_2D or n >= self.Pg_level_2D:
                        continue
                    else:
                        indexes_2D_out, points_n_out = self.fetch_2D_batches(binary_vxl_2D_list[axis_dim], n)  # [bs*(T+2)*(T+2)]
                        indexes_sotred_2D_out, indices_2D_out = torch.sort(indexes_2D_out, descending=False, dim=0)
                        # points_n_out = torch.index_select(points_n_out, dim=0, index=indices_2D_out)  # [bs*(T+2)*(T+2), n_features]
                        unique_value_2D_out, unique_cnt_2D_out = torch.unique(indexes_sotred_2D_out, return_counts=True)
                        unique_value_2D_out = unique_value_2D_out.to(torch.long)
                        batched_inputs_list[axis_dim].append([points_n_out, indices_2D_out, unique_value_2D_out + self.offsets_list_2D[n], unique_cnt_2D_out])
            self.batched_inputs_list = batched_inputs_list
        else:
            batched_inputs_list = self.batched_inputs_list

        for (Ec, p, b_2D, a, b_info) in zip([Encoding_xy, Encoding_xz, Encoding_yz], [params_q_xy, params_q_xz, params_q_yz], binary_vxl_2D_list, ['xy', 'xz', 'yz'], batched_inputs_list):

            ttl_bit_n, ttl_num_n = forward_2D(params_q_xyz[self.offsets_list[-2]:self.offsets_list[-1]], idx_coords2, Ec, p, b_2D, a, b_info)
            ttl_bit_sum += ttl_bit_n
            ttl_num_sum += ttl_num_n

        if sample_num is not None:
            sample_num_levels = torch.round((self.hashparams_num_levels * (sample_num / self.hashparams_num_levels.sum()))).to(torch.long)
            sample_num_levels = self.hashparams_num_levels if sample_num_levels[-1] > self.hashparams_num_levels[-1] else sample_num_levels
            print('sample_num_levels in forward: ', sample_num_levels)
            ttl_sample_num_valid_levels = 0
            for n in range(self.n_levels):
                if n not in self.skip_levels_3D and n < self.Pg_level:
                    ttl_sample_num_valid_levels += sample_num_levels[n].item()
        else:
            sample_num_levels = self.sample_num_levels
            ttl_sample_num_valid_levels = self.ttl_sample_num_valid_levels

        sample_start_levels_value_idx = (torch.round((self.hashparams_num_levels - sample_num_levels) * torch.rand_like(self.utils_rand))).to(torch.long)  # [n_levels]
        sample_end_levels_value_idx = sample_start_levels_value_idx + sample_num_levels
        sample_start_levels_pts_idx = self.unique_count_cumsum_list[self.utils_nlevel_idx, sample_start_levels_value_idx]
        sample_end_levels_pts_idx = self.unique_count_cumsum_list[self.utils_nlevel_idx, sample_end_levels_value_idx]
        #

        points_n_orig_list = []
        points_n_list = []
        Pg_n_list = []
        n_list = []
        unique_cnt_list = []
        values_q_list = []

        for n in range(self.n_levels):
            Pg_n, ttl_bit_n, ttl_num_n = self.get_BiRF_wentropy_leveln(params_q_xyz, n)
            if n in self.skip_levels_3D or n >= self.Pg_level:
                ttl_bit_sum += ttl_bit_n
            else:
                points_n_orig = self.pos_grid_sorted_list[n][sample_start_levels_pts_idx[n]:sample_end_levels_pts_idx[n]]
                points_n = (points_n_orig - 0.5) / self.scales_list[n, :]
                points_n_orig_list.append(points_n_orig)
                points_n_list.append(points_n)
                Pg_n_list.append(Pg_n.unsqueeze(-1).unsqueeze(-1).repeat(points_n.shape[0], 1))
                n_list.append(TORCH_N_LIST[n].repeat(points_n.shape[0]))
                unique_cnt_list.append(self.unique_count_list[n, sample_start_levels_value_idx[n]:sample_end_levels_value_idx[n]])
                hash_indices = self.unique_value_list[n][sample_start_levels_value_idx[n]:sample_end_levels_value_idx[n]] + self.offsets_list[n]
                values_q_list.append(params_q_xyz[hash_indices])

        if len(points_n_orig_list) > 0:
            points_n_orig_list = torch.cat(points_n_orig_list, dim=0)
            points_n_list = torch.cat(points_n_list, dim=0)
            Pg_n_list = torch.cat(Pg_n_list, dim=0)
            n_list = torch.cat(n_list, dim=0)
            unique_cnt_list = torch.cat(unique_cnt_list, dim=0)
            values_q_list = torch.cat(values_q_list, dim=0)

            mask, overlap_area_pool = self.query_binary_vxl_qlist(points_n_orig_list, binary_vxl, n_list, return_overlap_area=True)  # [points_num_levels[n], 1]
            mask_packed = align_and_pack.apply(mask.unsqueeze(-1).to(torch.float), unique_cnt_list, 0)  # [unique_cnt.numel(), unique_cnt.max(), 1]
            mask_cnt = (torch.sum(mask_packed[:, :, 0], dim=1)).to(torch.long)
            mask_exist = (torch.sum(mask_packed[:, :, 0], dim=1) > 0).to(torch.bool)

            points_n_list_masked = points_n_list[mask]
            n_list_masked = n_list[mask].to(torch.int)
            Pg_n_list_masked = Pg_n_list[mask]
            overlap_area_pool = overlap_area_pool[mask]

            mask_cnt = mask_cnt[mask_exist]  # [num_valid]
            values_q_list = values_q_list[mask_exist]  # [num_valid, n_features]

            overlap_area_pool = torch.clamp(overlap_area_pool, min=1)
            overlap_area_pool_packed = align_and_pack.apply(overlap_area_pool.unsqueeze(-1).to(torch.float), mask_cnt, 0)  # [num_valid, unique_cnt.max(), 1]
            overlap_area_pool_packed = overlap_area_pool_packed / torch.sum(overlap_area_pool_packed, dim=1, keepdim=True)  # [num_valid, unique_cnt.max(), 1]

            context_layer_num = self.max_context_layer_num
            context = Encoding_xyz.forward_diff_levels(points_n_list_masked, n_list_masked - context_layer_num, context_layer_num, binary_vxl=binary_vxl.squeeze(), PV=1001)
            context = torch.cat([context, Pg_n_list_masked], dim=-1)  # [num_valid, 4*3+1]
            mean = self.context_model_3D(context)  # [num_valid, 4]
            mean = align_and_pack.apply(mean, mask_cnt, 0.0)  # [num_valid, unique_cnt.max(), n_features]

            if self.use_overlap_area_pool:
                mean = mean * overlap_area_pool_packed
                mean = torch.sum(mean, dim=1)  # [num_valid, n_features]
            else:
                mean = torch.sum(mean, dim=1)  # [num_valid, n_features]
                mean = mean / mask_cnt.unsqueeze(-1)  # [num_valid, n_features]

            params_bit_n = self.entropy_model.forward(values_q_list, mean)  # [num_valid, n_features]
            ttl_bit_valid_levels = torch.sum(params_bit_n)

            ttl_bit_valid_levels = ttl_bit_valid_levels / ttl_sample_num_valid_levels * self.ttl_hashparams_num_valid_levels
            ttl_bit_sum += ttl_bit_valid_levels

        ttl_num_sum += params_q_xyz.numel()
        bits_per_param = ttl_bit_sum / ttl_num_sum

        return bits_per_param, ttl_bit_sum.item() / 8 / 1024 / 1024


    def encode_binary_vxl_mixPg_3D2D(self, Encoding_xyz, Encoding_xy, Encoding_xz, Encoding_yz, binary_vxl=None, filename_prefix='b'):
        Pgs_dict = {}
        def forward_2D(params_q_3D_clip, idx_coords2, Encoding_2D, params_q_2D, binary_vxl_2D, axis):
            if self.use_dimension_wise:
                pn_embed_frac = self.get_pn_embed_frac(params_q_3D_clip, idx_coords2, axis=axis)  # [514*514, 4]
            ttl_bit = 0
            encode_ttl_bit = 0
            for n in range(self.n_levels_2D):
                Pg_n, ttl_bit_n, ttl_num_n = self.get_BiRF_wentropy_leveln(params_q_2D, n, self.offsets_list_2D)
                Pgs_dict[axis + str(n)] = Pg_n
                if n in self.skip_levels_2D or n >= self.Pg_level_2D:
                    xs = params_q_2D[self.offsets_list_2D[n]:self.offsets_list_2D[n + 1]].view(-1)
                    ps = (Pg_n.unsqueeze(-1).unsqueeze(-1).repeat(self.offsets_list_2D[n+1]-self.offsets_list_2D[n], self.n_features)).view(-1)
                    encode_len_bit = encoder(xs, ps, filename_prefix + '_' + axis + str(n) +'.b')
                else:
                    indexes_2D, points_n = self.fetch_2D_batches(binary_vxl_2D, n)  # [bs*(T+2)*(T+2)]
                    context_layer_num = min(n, self.max_context_layer_num)

                    context = Encoding_2D(points_n, n-context_layer_num, n, binary_vxl=binary_vxl_2D, PV=0)  # [bs*(T+2)*(T+2), n_features*context_layer_num]
                    Pg_n = Pg_n.unsqueeze(-1).unsqueeze(-1).repeat(context.shape[0], 1)  # [N, 1]
                    if self.use_dimension_wise:
                        context_pn = Encoding_2D.forward_given_params(points_n,
                                                                      self.pn_frac_offsets_list,
                                                                      self.pn_frac_resolutions_list,
                                                                      pn_embed_frac, binary_vxl_2D).detach()  # [bs*(T+2)*(T+2), n_features]
                        context = torch.cat([context, context_pn, Pg_n], dim=-1)
                    else:
                        context = torch.cat([context, Pg_n], dim=-1)
                    mean = self.context_model_2D[n - 1](context)  # [bs*(T+2)*(T+2), n_features*2]

                    indexes_sotred_2D, indices_2D = torch.sort(indexes_2D, descending=False, dim=0)
                    unique_value_2D, unique_cnt_2D = torch.unique(indexes_sotred_2D, return_counts=True)
                    unique_value_2D = unique_value_2D.to(torch.long)

                    mean = torch.index_select(mean, dim=0, index=indices_2D)  # [bs*(T+2)*(T+2), n_features]
                    mean = align_and_pack.apply(mean, unique_cnt_2D, 0.0, 2)  # [num_valid, unique_cnt_2D.max(), n_features]
                    mean = torch.sum(mean, dim=1)  # [num_valid, n_features]
                    mean = mean / unique_cnt_2D.unsqueeze(-1)  # [num_valid, n_features]
                    values_q = params_q_2D[unique_value_2D + self.offsets_list_2D[n], :]  # [unique(bs*(T+2)*(T+2)), n_features]

                    params_bit_n = self.entropy_model.forward(values_q, mean)
                    xs = values_q.view(-1)
                    ps = (torch.clamp(mean, min=1e-6, max=1 - 1e-6)).view(-1)
                    encode_len_bit = encoder(xs, ps, filename_prefix + '_' + axis + str(n) + '.b')
                    ttl_bit_n = torch.sum(params_bit_n)
                ttl_bit += ttl_bit_n
                encode_ttl_bit += encode_len_bit

            ttl_num = params_q_2D.numel()
            return ttl_bit, ttl_num, encode_ttl_bit

        params_q_xy = self.get_STE_params(Encoding_xy)
        params_q_xz = self.get_STE_params(Encoding_xz)
        params_q_yz = self.get_STE_params(Encoding_yz)
        params_q_xyz = self.get_STE_params(Encoding_xyz)

        ttl_bit_sum = 0
        ttl_num_sum = 0
        encode_ttl_bit_sum = 0

        idx_coords2 = self.get_idx_coords2(binary_vxl)
        for (Ec, p, a) in zip([Encoding_xy, Encoding_xz, Encoding_yz], [params_q_xy, params_q_xz, params_q_yz], ['xy', 'xz', 'yz']):
            if a == 'xy':
                binary_vxl_2D = torch.any(binary_vxl.squeeze(0), dim=2)  # [Rb, Rb]
            elif a == 'xz':
                binary_vxl_2D = torch.any(binary_vxl.squeeze(0), dim=1)  # [Rb, Rb]
            elif a == 'yz':
                binary_vxl_2D = torch.any(binary_vxl.squeeze(0), dim=0)  # [Rb, Rb]
            else:
                raise NotImplementedError
            ttl_bit_n, ttl_num_n, encode_ttl_bit_n = forward_2D(params_q_xyz[self.offsets_list[-2]:self.offsets_list[-1]], idx_coords2, Ec, p, binary_vxl_2D, a)
            ttl_bit_sum += ttl_bit_n
            ttl_num_sum += ttl_num_n
            encode_ttl_bit_sum += encode_ttl_bit_n

        # 3D
        for n in range(self.n_levels):

            Pg_n, ttl_bit_n, ttl_num_n = self.get_BiRF_wentropy_leveln(params_q_xyz, n)
            Pgs_dict['3D' + str(n)] = Pg_n

            if n in self.skip_levels_3D or n >= self.Pg_level:
                ps = (Pg_n.unsqueeze(-1).unsqueeze(-1).repeat(self.offsets_list[n+1]-self.offsets_list[n], self.n_features)).view(-1)
                xs = params_q_xyz[self.offsets_list[n]:self.offsets_list[n + 1]].view(-1)
                encode_len_bit_n = encoder(xs, ps, filename_prefix + '_' + '3D' + str(n) + '.b')
                ttl_bit_sum += ttl_bit_n
                encode_ttl_bit_sum += encode_len_bit_n

            else:
                MAX_POINTS_NUM_TO_OOM = self.MAX_POINTS_NUM_TO_OOM  # 单次跑时允许的最多点数，为了防止OOM
                max_sample_num_level_n = int(MAX_POINTS_NUM_TO_OOM // self.utils_points_per_param_levels[n])
                hashparams_num_levels_n = int(self.hashparams_num_levels[n].item())
                sample_num_level_n = min(max_sample_num_level_n, hashparams_num_levels_n)
                sample_steps = int(np.ceil(hashparams_num_levels_n / sample_num_level_n))

                for sn in range(sample_steps):

                    sample_start_levels_value_idx_n = sn * sample_num_level_n
                    sample_end_levels_value_idx_n = min((sn + 1) * sample_num_level_n, hashparams_num_levels_n)
                    sample_start_levels_pts_idx_n = self.unique_count_cumsum_list[n, sample_start_levels_value_idx_n]
                    sample_end_levels_pts_idx_n = self.unique_count_cumsum_list[n, sample_end_levels_value_idx_n]

                    points_n_orig = self.pos_grid_sorted_list[n][sample_start_levels_pts_idx_n:sample_end_levels_pts_idx_n]
                    points_n = (points_n_orig - 0.5) / self.scales_list[n, :]
                    mask, overlap_area_pool = self.query_binary_vxl(points_n_orig, binary_vxl, n, return_overlap_area=True)  # [points_num_levels[n], 1]
                    # mask = mask*0 + 1
                    points_n = points_n[mask]
                    # unique_cnt_cumsum = self.unique_count_cumsum_list[n, sample_start_levels_value_idx_n:sample_end_levels_value_idx_n+1]
                    # unique_cnt = torch.diff(unique_cnt_cumsum, dim=0)

                    unique_cnt = self.unique_count_list[n, sample_start_levels_value_idx_n:sample_end_levels_value_idx_n]

                    mask_packed = align_and_pack.apply(mask.unsqueeze(-1).to(torch.float), unique_cnt, 0)  # [unique_cnt.numel(), unique_cnt.max(), 1]
                    mask_cnt = (torch.sum(mask_packed[:, :, 0], dim=1)).to(torch.long)  # [unique_cnt.numel()]
                    mask_exist = (torch.sum(mask_packed[:, :, 0], dim=1) > 0)  # [unique_cnt.numel()]
                    mask_cnt = mask_cnt[mask_exist]  # [num_valid] 用来取代 unique_cnt

                    overlap_area_pool = overlap_area_pool[mask]
                    overlap_area_pool = torch.clamp(overlap_area_pool, min=1)
                    overlap_area_pool_packed = align_and_pack.apply(overlap_area_pool.unsqueeze(-1).to(torch.float), mask_cnt, 0)  # [num_valid, unique_cnt.max(), 1]
                    overlap_area_pool_packed = overlap_area_pool_packed / torch.sum(overlap_area_pool_packed, dim=1, keepdim=True)  # [num_valid, unique_cnt.max(), 1]

                    hash_indices_n = self.unique_value_list[n][sample_start_levels_value_idx_n:sample_end_levels_value_idx_n] + self.offsets_list[n]
                    values_q = params_q_xyz[hash_indices_n]
                    values_q = values_q[mask_exist]  # [num_valid, n_features]

                    context_layer_num = min(n, self.max_context_layer_num)
                    context = Encoding_xyz(points_n, n - context_layer_num, n, outspace_params=None, binary_vxl=binary_vxl.squeeze(), PV=0)
                    Pg_n_tmp = Pg_n.unsqueeze(-1).unsqueeze(-1).repeat(context.shape[0], 1)  # [N, 1]
                    context = torch.cat([context, Pg_n_tmp], dim=-1)

                    mean = self.context_model_3D(context)

                    mean = align_and_pack.apply(mean, mask_cnt, 0.0)  # [num_valid, unique_cnt.max(), n_features]
                    if self.use_overlap_area_pool:
                        mean = mean * overlap_area_pool_packed
                        mean = torch.sum(mean, dim=1)  # [num_valid, n_features]
                    else:
                        mean = torch.sum(mean, dim=1)  # [num_valid, n_features]
                        mean = mean / mask_cnt.unsqueeze(-1)  # [num_valid, n_features]

                    params_bit_n = self.entropy_model.forward(values_q, mean)  # [points_num_levels[n], n_features]

                    ps = (torch.clamp(mean, min=1e-6, max=1 - 1e-6)).view(-1)
                    xs = values_q.view(-1)
                    encode_len_bit_n = encoder(xs, ps, filename_prefix + '_' + '3D' + str(n) + '_' + str(sn) + '.b')
                    ttl_bit_n = torch.sum(params_bit_n)
                    ttl_bit_sum += ttl_bit_n
                    encode_ttl_bit_sum += encode_len_bit_n

                    del points_n
                    del mask
                    del mask_packed
                    del mean
                    torch.cuda.empty_cache()

        return Pgs_dict, ttl_bit_sum.item()/8.0/1024/1024, encode_ttl_bit_sum/8.0/1024/1024

    def decode_binary_vxl_mixPg_3D2D(self, Encoding_xyz, Encoding_xy, Encoding_xz, Encoding_yz,
                                     params_q_xyz_rec, params_q_xy_rec, params_q_xz_rec, params_q_yz_rec, binary_vxl=None, Pgs_dict = None, filename_prefix='b'):
        def forward_2D(params_q_3D_clip, idx_coords2, Encoding_2D, params_q_2D_rec, binary_vxl_2D, axis):
            if self.use_dimension_wise:
                pn_embed_frac = self.get_pn_embed_frac(params_q_3D_clip, idx_coords2, axis=axis)  # [514*514, 4]
            for n in range(self.n_levels_2D):
                if n in self.skip_levels_2D or n >= self.Pg_level_2D:
                    Pg_n = Pgs_dict[axis+str(n)]
                    ps = (Pg_n.unsqueeze(-1).unsqueeze(-1).repeat(self.offsets_list_2D[n+1]-self.offsets_list_2D[n], self.n_features)).view(-1)
                    sout = decoder(ps, filename_prefix + '_' + axis + str(n) +'.b')
                    sout = sout.view(self.offsets_list_2D[n+1]-self.offsets_list_2D[n], self.n_features)
                    params_q_2D_rec[self.offsets_list_2D[n]: self.offsets_list_2D[n+1]] = sout
                else:
                    Pg_n = Pgs_dict[axis + str(n)]
                    indexes_2D, points_n = self.fetch_2D_batches(binary_vxl_2D, n)  # [bs*(T+2)*(T+2)]
                    context_layer_num = min(n, self.max_context_layer_num)

                    context = Encoding_2D(points_n, n-context_layer_num, n,
                                          outspace_params=params_q_2D_rec,
                                          binary_vxl=binary_vxl_2D, PV=0)  # [bs*(T+2)*(T+2), n_features*context_layer_num]
                    Pg_n = Pg_n.unsqueeze(-1).unsqueeze(-1).repeat(context.shape[0], 1)  # [N, 1]
                    if self.use_dimension_wise:
                        context_pn = Encoding_2D.forward_given_params(points_n,
                                                                      self.pn_frac_offsets_list,
                                                                      self.pn_frac_resolutions_list,
                                                                      pn_embed_frac, binary_vxl_2D).detach()  # [bs*(T+2)*(T+2), n_features]
                        context = torch.cat([context, context_pn, Pg_n], dim=-1)
                    else:
                        context = torch.cat([context, Pg_n], dim=-1)
                    mean = self.context_model_2D[n - 1](context)  # [bs*(T+2)*(T+2), n_features*2]

                    indexes_sotred_2D, indices_2D = torch.sort(indexes_2D, descending=False, dim=0)
                    unique_value_2D, unique_cnt_2D = torch.unique(indexes_sotred_2D, return_counts=True)
                    unique_value_2D = unique_value_2D.to(torch.long)

                    mean = torch.index_select(mean, dim=0, index=indices_2D)  # [bs*(T+2)*(T+2), n_features]
                    mean = align_and_pack.apply(mean, unique_cnt_2D, 0.0, 2)  # [num_valid, unique_cnt_2D.max(), n_features]
                    mean = torch.sum(mean, dim=1)  # [num_valid, n_features]
                    mean = mean / unique_cnt_2D.unsqueeze(-1)  # [num_valid, n_features]

                    ps = (torch.clamp(mean, min=1e-6, max=1 - 1e-6)).view(-1)
                    sout = decoder(ps, filename_prefix + '_' + axis + str(n) +'.b')
                    sout = sout.view(-1, self.n_features)
                    params_q_2D_rec[unique_value_2D + self.offsets_list_2D[n], :] = sout

        # 3D
        for n in range(self.n_levels):

            Pg_n = Pgs_dict['3D'+str(n)]

            if n in self.skip_levels_3D or n >= self.Pg_level:
                ps = (Pg_n.unsqueeze(-1).unsqueeze(-1).repeat(self.offsets_list[n+1]-self.offsets_list[n], self.n_features)).view(-1)
                sout = decoder(ps, filename_prefix + '_' + '3D' + str(n) + '.b')
                sout = sout.view(self.offsets_list[n+1]-self.offsets_list[n], self.n_features)
                params_q_xyz_rec[self.offsets_list[n]:self.offsets_list[n + 1]] = sout

            else:
                MAX_POINTS_NUM_TO_OOM = self.MAX_POINTS_NUM_TO_OOM
                max_sample_num_level_n = int(MAX_POINTS_NUM_TO_OOM // self.utils_points_per_param_levels[n])
                hashparams_num_levels_n = int(self.hashparams_num_levels[n].item())
                sample_num_level_n = min(max_sample_num_level_n, hashparams_num_levels_n)
                sample_steps = int(np.ceil(hashparams_num_levels_n / sample_num_level_n))

                for sn in range(sample_steps):

                    sample_start_levels_value_idx_n = sn * sample_num_level_n
                    sample_end_levels_value_idx_n = min((sn + 1) * sample_num_level_n, hashparams_num_levels_n)
                    sample_start_levels_pts_idx_n = self.unique_count_cumsum_list[n, sample_start_levels_value_idx_n]
                    sample_end_levels_pts_idx_n = self.unique_count_cumsum_list[n, sample_end_levels_value_idx_n]

                    points_n_orig = self.pos_grid_sorted_list[n][sample_start_levels_pts_idx_n:sample_end_levels_pts_idx_n]
                    points_n = (points_n_orig - 0.5) / self.scales_list[n, :]
                    mask, overlap_area_pool = self.query_binary_vxl(points_n_orig, binary_vxl, n, return_overlap_area=True)  # [points_num_levels[n], 1]

                    points_n = points_n[mask]

                    unique_cnt = self.unique_count_list[n, sample_start_levels_value_idx_n:sample_end_levels_value_idx_n]

                    mask_packed = align_and_pack.apply(mask.unsqueeze(-1).to(torch.float), unique_cnt, 0)  # [unique_cnt.numel(), unique_cnt.max(), 1]
                    mask_cnt = (torch.sum(mask_packed[:, :, 0], dim=1)).to(torch.long)  # [unique_cnt.numel()]
                    mask_exist = (torch.sum(mask_packed[:, :, 0], dim=1) > 0)  # [unique_cnt.numel()]
                    mask_cnt = mask_cnt[mask_exist]  # [num_valid] 用来取代 unique_cnt

                    overlap_area_pool = overlap_area_pool[mask]
                    overlap_area_pool = torch.clamp(overlap_area_pool, min=1)
                    overlap_area_pool_packed = align_and_pack.apply(overlap_area_pool.unsqueeze(-1).to(torch.float), mask_cnt, 0)  # [num_valid, unique_cnt.max(), 1]
                    overlap_area_pool_packed = overlap_area_pool_packed / torch.sum(overlap_area_pool_packed, dim=1, keepdim=True)  # [num_valid, unique_cnt.max(), 1]

                    context_layer_num = min(n, self.max_context_layer_num)
                    context = Encoding_xyz(points_n, n - context_layer_num, n, outspace_params=params_q_xyz_rec, binary_vxl=binary_vxl.squeeze(), PV=0)

                    Pg_n_tmp = Pg_n.unsqueeze(-1).unsqueeze(-1).repeat(context.shape[0], 1)  # [N, 1]
                    context = torch.cat([context, Pg_n_tmp], dim=-1)

                    mean = self.context_model_3D(context)

                    mean = align_and_pack.apply(mean, mask_cnt, 0.0)  # [num_valid, unique_cnt.max(), n_features]
                    if self.use_overlap_area_pool:
                        mean = mean * overlap_area_pool_packed
                        mean = torch.sum(mean, dim=1)  # [num_valid, n_features]
                    else:
                        mean = torch.sum(mean, dim=1)  # [num_valid, n_features]
                        mean = mean / mask_cnt.unsqueeze(-1)  # [num_valid, n_features]

                    mean_shape = mean.shape
                    ps = (torch.clamp(mean, min=1e-6, max=1 - 1e-6)).view(-1)
                    sout = decoder(ps, filename_prefix + '_' + '3D' + str(n) + '_' + str(sn) + '.b')
                    sout = sout.view(*mean_shape)

                    hash_indices_n = self.unique_value_list[n][sample_start_levels_value_idx_n:sample_end_levels_value_idx_n] + self.offsets_list[n]
                    hash_indices_n_masked = hash_indices_n[mask_exist]
                    params_q_xyz_rec[hash_indices_n_masked] = sout

                    del points_n
                    del mask
                    del mask_packed
                    del mean
                    torch.cuda.empty_cache()

        idx_coords2 = self.get_idx_coords2(binary_vxl)
        for (Ec, p, a) in zip([Encoding_xy, Encoding_xz, Encoding_yz], [params_q_xy_rec, params_q_xz_rec, params_q_yz_rec], ['xy', 'xz', 'yz']):
            if a == 'xy':
                binary_vxl_2D = torch.any(binary_vxl.squeeze(0), dim=2)  # [Rb, Rb]
            elif a == 'xz':
                binary_vxl_2D = torch.any(binary_vxl.squeeze(0), dim=1)  # [Rb, Rb]
            elif a == 'yz':
                binary_vxl_2D = torch.any(binary_vxl.squeeze(0), dim=0)  # [Rb, Rb]
            else:
                raise NotImplementedError
            forward_2D(params_q_xyz_rec[self.offsets_list[-2]:self.offsets_list[-1]], idx_coords2, Ec, p, binary_vxl_2D, a)


        return params_q_xyz_rec, params_q_xy_rec, params_q_xz_rec, params_q_yz_rec


class Bernoulli_entropy(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, p):
        # p = torch.sigmoid(p)
        p = torch.clamp(p, min=1e-6, max=1 - 1e-6)
        pos_mask = (1 + x) / 2.0  # 1 -> 1, -1 -> 0
        neg_mask = (1 - x) / 2.0  # -1 -> 1, 1 -> 0
        pos_prob = p
        neg_prob = 1 - p
        param_bit = -torch.log2(pos_prob) * pos_mask + -torch.log2(neg_prob) * neg_mask
        return param_bit


from typing import Callable, List, Union
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

import _gridencoder as _backend

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()

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

class _grid_encode(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, embeddings, offsets_list, resolutions_list, calc_grad_inputs=False, min_level_id=None, n_levels_calc=1, binary_vxl=None, PV=0):
        # inputs: [N, num_dim], float in [0, 1]
        # embeddings: [sO, n_features], float. self.params = nn.Parameter(torch.empty(offset, n_features))
        # offsets_list: [n_levels + 1], int
        # RETURN: [N, F], float
        inputs = inputs.contiguous()
        if calc_grad_inputs == True:
            print('calc_grad_inputs not applicable!')
            assert False

        Rb = 128
        if binary_vxl is not None:
            binary_vxl = binary_vxl.contiguous()
            Rb = binary_vxl.shape[-1]
            assert len(binary_vxl.shape) == inputs.shape[-1]

        N, num_dim = inputs.shape
        n_levels = offsets_list.shape[0] - 1
        n_features = embeddings.shape[1]

        max_level_id = min_level_id + n_levels_calc

        # manually handle autocast (only use half precision embeddings, inputs must be float for enough precision)
        # if n_features % 2 != 0, force float, since half for atomicAdd is very slow.
        if torch.is_autocast_enabled() and n_features % 2 == 0:
            embeddings = embeddings.to(torch.half)

        # n_levels first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.empty(n_levels_calc, N, n_features, device=inputs.device, dtype=embeddings.dtype)

        # zero init if we only calculate partial levels
        # if n_levels_calc < n_levels: outputs.zero_()
        dy_dx = None

        if isinstance(min_level_id, int):
            _backend.grid_encode_forward(
                inputs,
                embeddings,
                offsets_list[min_level_id:max_level_id+1],
                resolutions_list[min_level_id:max_level_id],
                outputs,
                N, num_dim, n_features, n_levels_calc, 0, Rb, PV,
                dy_dx,
                binary_vxl,
                None
                )
        else:
            _backend.grid_encode_forward(
                inputs,
                embeddings,
                offsets_list,
                resolutions_list,
                outputs,
                N, num_dim, n_features, n_levels_calc, 0, Rb, PV,
                dy_dx,
                binary_vxl,
                min_level_id
                )

        outputs = outputs.permute(1, 0, 2).reshape(N, n_levels_calc * n_features)

        ctx.save_for_backward(inputs, embeddings, offsets_list, resolutions_list, dy_dx, binary_vxl)
        ctx.dims = [N, num_dim, n_features, n_levels_calc, min_level_id, max_level_id, Rb, PV]

        return outputs

    @staticmethod
    #@once_differentiable
    @custom_bwd
    def backward(ctx, grad):

        inputs, embeddings, offsets_list, resolutions_list, dy_dx, binary_vxl = ctx.saved_tensors
        N, num_dim, n_features, n_levels_calc, min_level_id, max_level_id, Rb, PV = ctx.dims

        # grad: [N, n_levels * n_features] --> [n_levels, N, n_features]
        grad = grad.view(N, n_levels_calc, n_features).permute(1, 0, 2).contiguous()

        grad_embeddings = torch.zeros_like(embeddings)

        grad_inputs = None

        if isinstance(min_level_id, int):
            _backend.grid_encode_backward(
                grad,
                inputs,
                embeddings,
                offsets_list[min_level_id:max_level_id+1],
                resolutions_list[min_level_id:max_level_id],
                grad_embeddings,
                N, num_dim, n_features, n_levels_calc, 0, Rb,
                dy_dx,
                grad_inputs,
                binary_vxl,
                None
                )
        else:
            _backend.grid_encode_backward(
                grad,
                inputs,
                embeddings,
                offsets_list,
                resolutions_list,
                grad_embeddings,
                N, num_dim, n_features, n_levels_calc, 0, Rb,
                dy_dx,
                grad_inputs,
                binary_vxl,
                min_level_id
                )

        if dy_dx is not None:
            grad_inputs = grad_inputs.to(inputs.dtype)

        return grad_inputs, grad_embeddings, None, None, None, None, None, None, None

grid_encode = _grid_encode.apply



class GridEncoder(nn.Module):
    def __init__(self,
                 num_dim=3,
                 n_features=2,
                 resolutions_list=(16, 23, 32, 46, 64, 92, 128, 184, 256, 368, 512, 736),
                 log2_hashmap_size=19,
                 ste_binary = False,
                 ste_multistep = False,
                 add_noise = False,
                 Q = 1
                 ):
        super().__init__()

        resolutions_list = torch.tensor(resolutions_list).to(torch.int)
        n_levels = resolutions_list.numel()

        self.num_dim = num_dim # coord dims, 2 or 3
        self.n_levels = n_levels # num levels, each level multiply resolution by 2
        self.n_features = n_features # encode channels per level
        self.log2_hashmap_size = log2_hashmap_size
        self.output_dim = n_levels * n_features
        self.ste_binary = ste_binary
        self.ste_multistep = ste_multistep
        self.add_noise = add_noise
        self.Q = Q

        # allocate parameters
        offsets_list = []  # 每层hashtable长度的cumsum
        offset = 0  # 用于统计所有层加起来一共需要多少长度的hashtable
        self.max_params = 2 ** log2_hashmap_size  # 按论文算法的每层的hashtable长度上限
        for i in range(n_levels):
            resolution = resolutions_list[i].item()
            params_in_level = min(self.max_params, resolution ** num_dim) # limit max number
            params_in_level = int(np.ceil(params_in_level / 8) * 8) # make divisible
            offsets_list.append(offset)
            offset += params_in_level
        offsets_list.append(offset)
        offsets_list = torch.from_numpy(np.array(offsets_list, dtype=np.int32))
        self.register_buffer('offsets_list', offsets_list)
        self.register_buffer('resolutions_list', resolutions_list)

        self.n_params = offsets_list[-1] * n_features  # 所有的params的数量

        # parameters
        self.params = nn.Parameter(torch.empty(offset, n_features))

        self.reset_parameters()

        self.n_output_dims = n_levels * n_features

    def reset_parameters(self):
        std = 1e-4
        self.params.data.uniform_(-std, std)

    def __repr__(self):
        return f"GridEncoder: num_dim={self.num_dim} n_levels={self.n_levels} n_features={self.n_features} resolution={self.base_resolution} -> {int(round(self.base_resolution * self.per_level_scale ** (self.n_levels - 1)))} per_level_scale={self.per_level_scale:.4f} params={tuple(self.params.shape)} gridtype={self.gridtype} align_corners={self.align_corners} interpolation={self.interpolation}"

    def forward(self, inputs, min_level_id=None, max_level_id=None, test_phase=False, outspace_params=None, binary_vxl=None, PV=0):
        # inputs: [..., num_dim], normalized real world positions in [-1, 1]
        # return: [..., n_levels * n_features]

        # min_level_id是包含的，max_level_id是不包含的: [min_level_id, max_level_id)

        #print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.num_dim)

        if outspace_params is not None:
            params = nn.Parameter(outspace_params)
        else:
            params = self.params

        if self.ste_binary:
            embeddings = STE_binary.apply(params)
            # embeddings = params
        elif (self.add_noise and not test_phase):
            embeddings = params + (torch.rand_like(params) - 0.5) * (1 / self.Q)
        elif (self.ste_multistep) or (self.add_noise and test_phase):
            embeddings = STE_multistep.apply(params, self.Q)
        else:
            embeddings = params

        min_level_id = 0 if min_level_id is None else max(min_level_id, 0)
        max_level_id = self.n_levels if max_level_id is None else min(max_level_id, self.n_levels)
        n_levels_calc = max_level_id - min_level_id

        outputs = grid_encode(inputs, embeddings, self.offsets_list, self.resolutions_list, False, min_level_id, n_levels_calc, binary_vxl, PV)
        outputs = outputs.view(prefix_shape + [n_levels_calc * self.n_features])

        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs

    def forward_diff_levels(self, inputs, min_level_id_list=None, n_levels_calc=1, test_phase=False, outspace_params=None, binary_vxl=None, PV=0):
        # inputs: [..., num_dim], normalized real world positions in [-1, 1]
        # return: [..., n_levels * n_features]

        # min_level_id是包含的，max_level_id是不包含的: [min_level_id, max_level_id)

        #print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.num_dim)

        if outspace_params is not None:
            params = nn.Parameter(outspace_params)
        else:
            params = self.params

        if self.ste_binary:
            embeddings = STE_binary.apply(params)
            # embeddings = params
        elif (self.add_noise and not test_phase):
            embeddings = params + (torch.rand_like(params) - 0.5) * (1 / self.Q)
        elif (self.ste_multistep) or (self.add_noise and test_phase):
            embeddings = STE_multistep.apply(params, self.Q)
        else:
            embeddings = params

        outputs = grid_encode(inputs, embeddings, self.offsets_list, self.resolutions_list, False, min_level_id_list.contiguous(), n_levels_calc, binary_vxl, PV)

        outputs = outputs.view(prefix_shape + [n_levels_calc * self.n_features])

        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs

    def forward_given_params(self, inputs, offsets_list, resolutions_list, outspace_params=None, binary_vxl=None, PV=0):

        assert inputs.shape[-1] == 2
        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, 2)

        embeddings = outspace_params

        min_level_id = 0
        max_level_id = 1
        n_levels_calc = max_level_id - min_level_id
        outputs = grid_encode(inputs, embeddings, offsets_list, resolutions_list, False, min_level_id, n_levels_calc, binary_vxl, PV)
        outputs = outputs.view(prefix_shape + [n_levels_calc * self.n_features])

        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def contract_to_unisphere(
    x: torch.Tensor,
    aabb: torch.Tensor,
    ord: Union[str, int] = 2,
    #  ord: Union[float, int] = float("inf"),
    eps: float = 1e-6,
    derivative: bool = False,
):
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1  # aabb is at [-1, 1]
    mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1

    if derivative:
        dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
            1 / mag**3 - (2 * mag - 1) / mag**4
        )
        dev[~mask] = 1.0
        dev = torch.clamp(dev, min=eps)
        return dev
    else:
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
        return x



class NGPRadianceField_mygrid_2D3D(torch.nn.Module):
    """Instance-NGP Radiance Field"""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        use_viewdirs: bool = True,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        geo_feat_dim: int = 15,

        resolutions_list = (16, 22, 31, 42, 57, 78, 106, 146, 199, 273, 374, 512),
        log2_hashmap_size: int = 19,

        resolutions_list_2D = (64, 128, 256, 512, 1024),
        log2_hashmap_size_2D = 17,

        n_features_per_level = 2,
        n_neurons = 64,
        ste_binary = True,
        ste_multistep = False,
        add_noise = False,
        Q = 10,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.unbounded = unbounded
        geo_feat_dim = n_features_per_level*10 - 1
        geo_feat_dim = max(15, geo_feat_dim)
        geo_feat_dim = min(127, geo_feat_dim)
        self.geo_feat_dim = geo_feat_dim

        self.resolutions_list = resolutions_list
        self.log2_hashmap_size = log2_hashmap_size

        self.resolutions_list_2D = resolutions_list_2D
        self.log2_hashmap_size_2D = log2_hashmap_size_2D

        # print('NeRF settings:', self.geo_feat_dim, self.log2_hashmap_size, self.log2_hashmap_size_2D)

        if self.use_viewdirs:
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=num_dim,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "SphericalHarmonics",
                            "degree": 4,
                        },
                        # {"otype": "Identity", "n_bins": 4, "degree": 4},
                    ],
                },
            )

        encoding_xyz = GridEncoder(
            num_dim=3,
            n_features=n_features_per_level,
            resolutions_list=resolutions_list,
            log2_hashmap_size=log2_hashmap_size,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        encoding_xy = GridEncoder(
            num_dim=2,
            n_features=n_features_per_level,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        encoding_xz = GridEncoder(
            num_dim=2,
            n_features=n_features_per_level,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        encoding_yz = GridEncoder(
            num_dim=2,
            n_features=n_features_per_level,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        embed_fn, input_ch = get_embedder(10, 0)

        the_input_chs = encoding_xyz.n_output_dims + encoding_xy.n_output_dims + \
                encoding_xz.n_output_dims + encoding_yz.n_output_dims + input_ch
        the_output_chs = 1 + self.geo_feat_dim

        network = nn.Sequential(
            nn.Linear(the_input_chs, n_neurons),
            nn.ReLU(inplace=True),
            nn.Linear(n_neurons, the_output_chs),
        )

        self.mlp_base = compose_3D_2D_embed(
            encoding_xyz,
            encoding_xy, encoding_xz, encoding_yz,
            embed_fn, network)

        if self.geo_feat_dim > 0:

            the_input_chs = (
                    (
                        self.direction_encoding.n_output_dims
                        if self.use_viewdirs
                        else 0
                    )
                    + self.geo_feat_dim
                )
            the_output_chs = 3

            self.mlp_head = nn.Sequential(
                nn.Linear(the_input_chs, n_neurons),
                nn.ReLU(inplace=True),
                nn.Linear(n_neurons, n_neurons),
                nn.ReLU(inplace=True),
                nn.Linear(n_neurons, the_output_chs),
            )


    def update_embedding_params(self, params_q_xyz_rec, params_q_xy_rec, params_q_xz_rec, params_q_yz_rec):
        self.mlp_base.encoding_xyz.params = nn.Parameter(params_q_xyz_rec)
        self.mlp_base.encoding_xy.params = nn.Parameter(params_q_xy_rec)
        self.mlp_base.encoding_xz.params = nn.Parameter(params_q_xz_rec)
        self.mlp_base.encoding_yz.params = nn.Parameter(params_q_yz_rec)
        print('embedding_params updated!')

    def query_density(self, x, return_feat: bool = False):
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        x = (
            self.mlp_base(x.view(-1, self.num_dim))
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )
        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )
        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )
        if return_feat:
            return density, base_mlp_out
        else:
            return density

    def _query_rgb(self, dir, embedding, apply_act: bool = True):
        if self.use_viewdirs:
            dir = (dir + 1.0) / 2.0
            d = self.direction_encoding(dir.reshape(-1, dir.shape[-1]))
            h = torch.cat([d, embedding.reshape(-1, self.geo_feat_dim)], dim=-1)
        else:
            h = embedding.reshape(-1, self.geo_feat_dim)
        rgb = (
            self.mlp_head(h)
            .reshape(list(embedding.shape[:-1]) + [3])
            .to(embedding)
        )
        if apply_act:
            rgb = torch.sigmoid(rgb)
        return rgb

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor = None,
    ):
        if self.use_viewdirs and (directions is not None):
            assert (
                positions.shape == directions.shape
            ), f"{positions.shape} v.s. {directions.shape}"

            density, embedding = self.query_density(positions, return_feat=True)
            rgb = self._query_rgb(directions, embedding=embedding)
        return rgb, density  # type: ignore


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class compose_3D_2D_embed(nn.Module):
    def __init__(self, encoding_xyz, encoding_xy, encoding_xz, encoding_yz, embed_fn, network, sin_encode=False):
        super().__init__()
        self.encoding_xyz = encoding_xyz
        self.encoding_xy = encoding_xy
        self.encoding_xz = encoding_xz
        self.encoding_yz = encoding_yz
        self.embed_fn = embed_fn
        self.network = network
    def forward(self, x):
        # x: [..., 3]
        x_x, y_y, z_z = torch.chunk(x, 3, dim=-1)

        out_xyz = self.encoding_xyz(x)  # [..., 2*16]

        out_xy = self.encoding_xy(torch.cat([x_x, y_y], dim=-1))  # [..., 2*4]
        out_xz = self.encoding_xz(torch.cat([x_x, z_z], dim=-1))  # [..., 2*4]
        out_yz = self.encoding_yz(torch.cat([y_y, z_z], dim=-1))  # [..., 2*4]
        out_i = torch.cat([out_xyz, out_xy, out_xz, out_yz], dim=-1)  # [..., 56]

        if self.embed_fn is not None:
            out_sine = self.embed_fn(x)
            out_i = torch.cat([out_i, out_sine], dim=-1)

        output = self.network(out_i)  # [..., 16]
        return output


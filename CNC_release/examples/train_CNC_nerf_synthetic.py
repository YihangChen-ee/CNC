import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import math
import pathlib
import time

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from lpips import LPIPS
from pytorch_ssim import ssim
from radiance_fields.ngp import NGPRadianceField_mygrid_2D3D

from utils import (
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    render_image_with_occgrid_test,
    set_random_seed,
)
from utils_bpp_acc import (
    CNC_context_models,
)
from nerfacc.estimators.occ_grid import OccGridEstimator


def quantize_params(dict_input, ss=None, digits=10):
    ss_cnt = 0
    bits = 0
    bits_orig = 0
    dict_quantized = {}
    quantized_v_list = []
    for n, p_input in dict_input.items():
        min_v = torch.min(p_input)
        max_v = torch.max(p_input)
        scales = 2 ** digits - 1
        interval = ((max_v - min_v) / scales + 1e-6)  # avoid 0, if max_v == min_v
        quantized_v = (p_input - min_v) // interval
        p_input_q = quantized_v * interval + min_v

        quantized_v_list.append(quantized_v)
        dict_quantized[n] = p_input_q
        bits += digits * p_input_q.numel() + 32 + 32  # + 32 + 32 for min_v and scale
        bits_orig += 32 * p_input_q.numel()
        ss_cnt += 1

    return bits/8.0/1024/1024, bits_orig/8.0/1024/1024, dict_quantized, quantized_v_list


def get_binary_vxl_size(binary_vxl):
    with torch.no_grad():
        ttl_num = binary_vxl.numel()

        pos_num = torch.sum(binary_vxl)
        neg_num = ttl_num - pos_num

        Pg = pos_num / ttl_num
        pos_prob = Pg
        neg_prob = (1 - Pg)
        pos_bit = pos_num * (-torch.log2(pos_prob))
        neg_bit = neg_num * (-torch.log2(neg_prob))
        ttl_bit = pos_bit + neg_bit
        ttl_bit += 32  # Pg
        # print('binary_vxl:', Pg, ttl_bit, ttl_num, pos_num, neg_num)
    return Pg, ttl_bit.item()/8.0/1024/1024, ttl_num


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    default=str(pathlib.Path.cwd() / "data/nerf_synthetic"),
    help="the root dir of the dataset",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
    help="which train split to use",
)
parser.add_argument(
    "--scene",
    type=str,
    # default="lego",
    default="chair",
    choices=NERF_SYNTHETIC_SCENES,
    help="which scene to use",
)
parser.add_argument(
    "--lmbda",
    type=float,
    default=2e-3,
)
parser.add_argument(
    "--Pg_level",
    type=int,
    default=12,
)
parser.add_argument(
    "--Pg_level_2D",
    type=int,
    default=4,
)
parser.add_argument(
    "--log2_hashmap_size",
    type=int,
    default=19,
)
parser.add_argument(
    "--log2_hashmap_size_2D",
    type=int,
    default=17,
)
parser.add_argument(
    "--sample_num",
    type=int,
    default=200000,
)
parser.add_argument(
    "--max_context_layer_num",
    type=int,
    default=3,
)
parser.add_argument(
    "--n_features",
    type=int,
    default=4,
)
args = parser.parse_args()

device = "cuda:0"
set_random_seed(42)

lmbda = args.lmbda
n_neurons = 160
sample_num = args.sample_num
max_context_layer_num = args.max_context_layer_num
ste_binary = True
ste_multistep = False
add_noise = False
Q = 10
Pg_level = args.Pg_level
Pg_level_2D = args.Pg_level_2D

entropoy_loss_invove_epoch = 0
resolutions_list = [16, 22, 31, 42, 57, 78, 106, 146, 199, 273, 374, 512]
resolutions_list = list(np.array(resolutions_list) + 2)
log2_hashmap_size = args.log2_hashmap_size
resolutions_list_2D = (128, 256, 512, 1024)
resolutions_list_2D = list(np.array(resolutions_list_2D) + 2)
log2_hashmap_size_2D = args.log2_hashmap_size_2D

step_update = 16
skip_levels_3D = [0, 1, 2]
skip_levels_2D = [0]

for n_features in [args.n_features]:
    for current_scene in [args.scene]:
        if 1:
            from datasets.nerf_synthetic import SubjectLoader

            # training parameters
            max_steps = 20000
            init_batch_size = 1024
            target_sample_batch_size = 1 << 18
            weight_decay = (
                2e-5 if current_scene in ["drums"] else 2e-6
            )
            # scene parameters
            aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
            near_plane = 0.0
            far_plane = 1.0e10
            # dataset parameters
            train_dataset_kwargs = {}
            test_dataset_kwargs = {}
            # model parameters
            grid_resolution = 128
            grid_nlvl = 1
            # render parameters
            render_step_size = 5e-3
            alpha_thre = 0.0
            cone_angle = 0.0

        train_dataset = SubjectLoader(
            subject_id=current_scene,
            root_fp=args.data_root,
            split=args.train_split,
            num_rays=init_batch_size,
            device=device,
            **train_dataset_kwargs,
        )

        test_dataset = SubjectLoader(
            subject_id=current_scene,
            root_fp=args.data_root,
            split="test",
            num_rays=None,
            device=device,
            **test_dataset_kwargs,
        )

        estimator = OccGridEstimator(
            roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
        ).to(device)

        # setup the radiance field we want to train.
        grad_scaler = torch.cuda.amp.GradScaler(2**10)
        # mlp_base.params torch.Size([12602992=12599920+3072]), mlp_head.params torch.Size([7168])
        radiance_field = NGPRadianceField_mygrid_2D3D(aabb=estimator.aabbs[-1],
                                          n_features_per_level=n_features,
                                          n_neurons=n_neurons,

                                          resolutions_list=resolutions_list,
                                          log2_hashmap_size=log2_hashmap_size,

                                          resolutions_list_2D=resolutions_list_2D,
                                          log2_hashmap_size_2D=log2_hashmap_size_2D,

                                          ste_binary=ste_binary,
                                          ste_multistep=ste_multistep,
                                          add_noise=add_noise,
                                          Q=Q,
                                          ).to(device)

        entropy_estimator = CNC_context_models(
            num_dim=3,
            resolutions_list=resolutions_list,
            resolutions_list_2D=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size,
            log2_hashmap_size_2D=log2_hashmap_size_2D,
            n_features=n_features,
            sample_num=sample_num,
            max_context_layer_num=max_context_layer_num,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
            Pg_level=Pg_level,
            Pg_level_2D=Pg_level_2D,
            step_update=step_update,
            skip_levels_3D=skip_levels_3D,
            skip_levels_2D=skip_levels_2D,
        ).cuda()

        context_MBs_orig = 0
        for (n, p) in entropy_estimator.named_parameters():
            context_MBs_orig += p.numel() * 32
        context_MBs_orig = context_MBs_orig / 8.0 / 1024 / 1024

        optimizer = torch.optim.Adam(
            [
                {'params': radiance_field.parameters()},
            ],
            lr=6e-3, eps=1e-15, weight_decay=weight_decay  # in paper: lr=1e-2
        )

        optimizer2 = torch.optim.Adam(
            [
                {'params': entropy_estimator.parameters()}  # in paper: lr=1e-2
            ],
            lr=6e-3, eps=1e-15
        )

        milestones_my = [9000, 12000, 15000, 17000, 19000]

        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, total_iters=1000
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=milestones_my,
                    gamma=0.33,
                ),
            ]
        )

        scheduler2 = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer2, start_factor=0.01, total_iters=1000
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    optimizer2,
                    milestones=milestones_my,
                    gamma=0.33,
                ),
            ]
        )

        lpips_net = LPIPS(net="vgg").to(device)
        lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
        lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

        # training
        tic = time.time()
        for step in range(max_steps + 1):

            radiance_field.train()
            estimator.train()

            i = torch.randint(0, len(train_dataset), (1,)).item()
            data = train_dataset[i]

            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]

            def occ_eval_fn(x):
                density = radiance_field.query_density(x)
                return density * render_step_size

            # update occupancy grid
            estimator.update_every_n_steps(
                step=step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=1e-2,
                n=step_update,
            )
            rgb, acc, depth, n_rendering_samples, extra = render_image_with_occgrid(
                radiance_field,
                estimator,
                rays,
                # rendering options
                near_plane=near_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=cone_angle,
                alpha_thre=alpha_thre,
                return_extra=True,
            )
            if n_rendering_samples == 0:
                continue

            if target_sample_batch_size > 0:
                # dynamic batch size for rays to keep sample batch size constant.
                num_rays = len(pixels)
                num_rays = int(num_rays * (target_sample_batch_size / float(n_rendering_samples)))
                train_dataset.update_num_rays(num_rays)

            loss = F.mse_loss(rgb, pixels)

            bits_per_param = 0
            embed_bits_MB = 0
            if step >= entropoy_loss_invove_epoch and lmbda>0:  # always true
                bits_per_param, embed_bits_MB = entropy_estimator.forward_binary_vxl_mixPg_3D2D(radiance_field.mlp_base.encoding_xyz,
                                                                                 radiance_field.mlp_base.encoding_xy,
                                                                                 radiance_field.mlp_base.encoding_xz,
                                                                                 radiance_field.mlp_base.encoding_yz,
                                                                                 estimator.binaries, verbose=step%1000==0, sample_num=None, step=step)
                loss = loss + lmbda*bits_per_param
                bits_per_param = bits_per_param.item()

            optimizer.zero_grad()
            if lmbda > 0: optimizer2.zero_grad()
            # do not unscale it because we are using Adam.
            grad_scaler.scale(loss).backward()
            optimizer.step()
            if lmbda > 0:optimizer2.step()
            scheduler.step()
            if lmbda > 0:scheduler2.step()

            if step % 200 == 0:

                elapsed_time = time.time() - tic
                loss = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(loss) / np.log(10.0)
                loss = loss + lmbda * bits_per_param
                print(
                    f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                    f"loss={loss:.5f} | psnr={psnr:.2f} | "
                    f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
                    f"max_depth={depth.max():.3f} | "
                    f"bits_per_param={bits_per_param:.3f} | "
                    f"embed_bits_MB={embed_bits_MB:.3f} | "
                )


            if step > 0 and step % max_steps == 0:
                # evaluation
                radiance_field.eval()
                estimator.eval()

                torch.cuda.empty_cache()

                psnrs = []
                lpips = []
                ssims = []
                with torch.no_grad():  # instant evaluation after training
                    for i in range(len(test_dataset)):
                        data = test_dataset[i]
                        render_bkgd = data["color_bkgd"]
                        rays = data["rays"]
                        pixels = data["pixels"]  # [800, 800, 3], 0~1

                        # rendering
                        rgb, acc, depth, _ = render_image_with_occgrid_test(
                            1024,
                            # scene
                            radiance_field,
                            estimator,
                            rays,
                            # rendering options
                            near_plane=near_plane,
                            render_step_size=render_step_size,
                            render_bkgd=render_bkgd,
                            cone_angle=cone_angle,
                            alpha_thre=alpha_thre,
                        )

                        mse = F.mse_loss(rgb, pixels)
                        psnr = -10.0 * torch.log(mse) / np.log(10.0)
                        psnrs.append(psnr.item())
                        lpips.append(lpips_fn(rgb, pixels).item())
                        ssims.append(-ssim(rgb.permute(2, 0, 1).unsqueeze(0), pixels.permute(2, 0, 1).unsqueeze(0)).item())
                psnr_avg = sum(psnrs) / len(psnrs)
                lpips_avg = sum(lpips) / len(lpips)
                ssim_avg = sum(ssims) / len(ssims)

                embed_bits_MB = 0
                embed_bits_MB_codec = 0
                encoding_time = 0
                decoding_time = 0

                if lmbda > 0:  # running the codec process
                    t1 = time.time()
                    with torch.no_grad():  # encoding
                        Pgs_dict, embed_bits_MB, embed_bits_MB_codec = entropy_estimator.encode_binary_vxl_mixPg_3D2D(
                            radiance_field.mlp_base.encoding_xyz,
                            radiance_field.mlp_base.encoding_xy,
                            radiance_field.mlp_base.encoding_xz,
                            radiance_field.mlp_base.encoding_yz,
                            estimator.binaries,
                            './bitstreams/' + args.scene
                        )
                    print(f"evaluation_orig: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}, "
                          f"ssim_avg={ssim_avg}, size={embed_bits_MB}")

                    # force all embeddings to zero and run embedding decoding
                    nn.init.constant_(radiance_field.mlp_base.encoding_xyz.params, 0)
                    nn.init.constant_(radiance_field.mlp_base.encoding_xy.params, 0)
                    nn.init.constant_(radiance_field.mlp_base.encoding_xz.params, 0)
                    nn.init.constant_(radiance_field.mlp_base.encoding_yz.params, 0)
                    t2 = time.time()
                    with torch.no_grad():  # decoding
                        params_q_xyz_rec, params_q_xy_rec, params_q_xz_rec, params_q_yz_rec = entropy_estimator.decode_binary_vxl_mixPg_3D2D(
                            radiance_field.mlp_base.encoding_xyz,
                            radiance_field.mlp_base.encoding_xy,
                            radiance_field.mlp_base.encoding_xz,
                            radiance_field.mlp_base.encoding_yz,
                            torch.ones_like(radiance_field.mlp_base.encoding_xyz.params),
                            torch.ones_like(radiance_field.mlp_base.encoding_xy.params),
                            torch.ones_like(radiance_field.mlp_base.encoding_xz.params),
                            torch.ones_like(radiance_field.mlp_base.encoding_yz.params),
                            estimator.binaries,
                            Pgs_dict,
                            './bitstreams/' + args.scene
                        )
                    t3 = time.time()
                    encoding_time = t2 - t1
                    decoding_time = t3 - t2
                    print('Encoded bitstreams saved to ./bitstreams/' + args.scene)
                    print('codec time:', encoding_time, decoding_time)
                    # update decoded embeddings to encodings
                    radiance_field.update_embedding_params(params_q_xyz_rec, params_q_xy_rec, params_q_xz_rec, params_q_yz_rec)

                # evaluate after codec
                psnrs_codec = []
                lpips_codec = []
                ssims_codec = []
                with torch.no_grad():
                    for i in range(len(test_dataset)):
                        data = test_dataset[i]
                        render_bkgd = data["color_bkgd"]
                        rays = data["rays"]
                        pixels = data["pixels"]
                        # rendering
                        rgb, acc, depth, _ = render_image_with_occgrid_test(
                            1024,
                            # scene
                            radiance_field,
                            estimator,
                            rays,
                            # rendering options
                            near_plane=near_plane,
                            render_step_size=render_step_size,
                            render_bkgd=render_bkgd,
                            cone_angle=cone_angle,
                            alpha_thre=alpha_thre,
                        )
                        mse = F.mse_loss(rgb, pixels)
                        psnr = -10.0 * torch.log(mse) / np.log(10.0)
                        psnrs_codec.append(psnr.item())
                        lpips_codec.append(lpips_fn(rgb, pixels).item())
                        ssims_codec.append(-ssim(rgb.permute(2, 0, 1).unsqueeze(0), pixels.permute(2, 0, 1).unsqueeze(0)).item())
                psnr_avg_codec = sum(psnrs_codec) / len(psnrs_codec)
                lpips_avg_codec = sum(lpips_codec) / len(lpips_codec)
                ssim_avg_codec = sum(ssims_codec) / len(ssims_codec)
                print(f"evaluation_decoded: psnr_avg_codec={psnr_avg_codec}, lpips_avg_codec={lpips_avg_codec}, "
                      f"ssim_avg_codec={ssim_avg_codec}, size_codec={embed_bits_MB_codec}")
                
                # quantize the rendering MLP
                orig_main_MLP_dict = {}
                for n, p in radiance_field.named_parameters():
                    if 'encoding' not in n:
                        orig_main_MLP_dict[n] = p
                digit_list = [13]
                main_MLP_MBs_list = []
                psnrs_list = []
                lpipses_list = []
                ssims_list = []
                for digit in digit_list:
                    MBs, MBs_orig, p_q_dict, v_list = quantize_params(orig_main_MLP_dict, digits=digit)
                    radiance_field.load_state_dict(state_dict=p_q_dict, strict=False)
                    psnrs_quantize = []
                    lpips_quantize = []
                    ssim_quantize = []
                    with torch.no_grad():
                        for i in range(len(test_dataset)):
                            data = test_dataset[i]
                            render_bkgd = data["color_bkgd"]
                            rays = data["rays"]
                            pixels = data["pixels"]
                            # rendering
                            rgb, acc, depth, _ = render_image_with_occgrid_test(
                                1024,
                                # scene
                                radiance_field,
                                estimator,
                                rays,
                                # rendering options
                                near_plane=near_plane,
                                render_step_size=render_step_size,
                                render_bkgd=render_bkgd,
                                cone_angle=cone_angle,
                                alpha_thre=alpha_thre,
                            )
                            mse = F.mse_loss(rgb, pixels)
                            psnr = -10.0 * torch.log(mse) / np.log(10.0)
                            psnrs_quantize.append(psnr.item())
                            lpips_quantize.append(lpips_fn(rgb, pixels).item())
                            ssim_quantize.append(-ssim(rgb.permute(2, 0, 1).unsqueeze(0), pixels.permute(2, 0, 1).unsqueeze(0)).item())
                    psnr_avg_quantize = sum(psnrs_quantize) / len(psnrs_quantize)
                    lpips_avg_quantize = sum(lpips_quantize) / len(lpips_quantize)
                    ssim_avg_quantize = sum(ssim_quantize) / len(ssim_quantize)

                    main_MLP_MBs_list.append(MBs)
                    psnrs_list.append(psnr_avg_quantize)
                    lpipses_list.append(lpips_avg_quantize)
                    ssims_list.append(ssim_avg_quantize)

                _, binary_vxl_MBs, _ = get_binary_vxl_size(estimator.binaries.view(-1))

                # print('main_MLP_size_orig in MB:', MBs_orig)

                if 1:
                    os.makedirs('./results/Synthetic-NeRF', exist_ok=True)
                    Q_step = '_binary' if ste_binary else ('_Q' + str(Q))
                    with open('./results/Synthetic-NeRF/output.txt', 'a') as fw:

                        fw.write(current_scene)
                        fw.write('\t')
                        fw.write(str(np.round(psnr_avg, decimals=4)))
                        fw.write('\t')
                        fw.write(str(np.round(lpips_avg, decimals=4)))
                        fw.write('\t')
                        fw.write(str(np.round(ssim_avg, decimals=4)))
                        fw.write('\t')
                        fw.write(str(np.round(psnr_avg_codec, decimals=4)))
                        fw.write('\t')
                        fw.write(str(np.round(lpips_avg_codec, decimals=4)))
                        fw.write('\t')
                        fw.write(str(np.round(ssim_avg_codec, decimals=4)))
                        fw.write('\t')
                        fw.write(str(np.round(embed_bits_MB, decimals=4)))
                        fw.write('\t')
                        fw.write(str(np.round(embed_bits_MB_codec, decimals=4)))
                        fw.write('\t')
                        fw.write(str(np.round(MBs_orig, decimals=4)))
                        fw.write('\t')
                        fw.write(str(np.round(context_MBs_orig, decimals=4)))
                        fw.write('\t')
                        fw.write(str(np.round(binary_vxl_MBs, decimals=4)))

                        for i in range(len(digit_list)):
                            fw.write('\t')
                            fw.write(str(digit_list[i]))
                            fw.write('\t')
                            fw.write(str(np.round(main_MLP_MBs_list[i], decimals=4)))
                            fw.write('\t')
                            fw.write(str(np.round(psnrs_list[i], decimals=4)))
                            fw.write('\t')
                            fw.write(str(np.round(lpipses_list[i], decimals=4)))
                            fw.write('\t')
                            fw.write(str(np.round(ssims_list[i], decimals=4)))

                        fw.write('\t')
                        fw.write(str(np.round(elapsed_time, decimals=4)))
                        fw.write('\t')
                        fw.write(str(np.round(encoding_time, decimals=4)))
                        fw.write('\t')
                        fw.write(str(np.round(decoding_time, decimals=4)))

                        fw.write('\n')

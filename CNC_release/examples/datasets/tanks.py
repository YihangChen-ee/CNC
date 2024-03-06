import torch
import torch.nn.functional as F
import json
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import imageio.v2 as imageio
from PIL import Image
from torchvision import transforms as T
import numpy as np
from .utils import Rays
from .ray_utils import *


def _load_renderings_NSVF(root_fp: str, subject_id: str, split: str):
    """Load images from disk."""
    if not root_fp.startswith("/"):
        # allow relative path. e.g., "./data/nerf_synthetic/"
        root_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            root_fp,
        )

    data_dir = os.path.join(root_fp, subject_id)

    id_map = {
        'train': '0_',
        'val': '1_',
        'test': '1_',
    }

    #
    rgb_files = [x for x in os.listdir(os.path.join(data_dir, 'rgb')) if x.startswith(id_map[split])]
    rgb_files.sort()  # 必须要sort，否则rgb_files和pose_files不匹配
    pose_files = [x for x in os.listdir(os.path.join(data_dir, 'pose')) if x.startswith(id_map[split])]
    pose_files.sort()

    images = []
    camtoworlds = []

    for i in range(len(rgb_files)):
        assert pose_files[i].split('.')[0].split('_')[-1] == rgb_files[i].split('.')[0].split('_')[-1]
        c2w = np.loadtxt(os.path.join(data_dir, 'pose', pose_files[i]))
        camtoworlds.append(c2w)
        rgba = imageio.imread(os.path.join(data_dir, 'rgb', rgb_files[i]))
        images.append(rgba)

    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)

    '''h, w = images.shape[1:3]
    with open(os.path.join(data_dir, "intrinsics.txt")) as f:
        focal = float(f.readline().split()[0])'''
    intrinsics = np.loadtxt(os.path.join(data_dir, "intrinsics.txt"))
    intrinsics = torch.from_numpy(intrinsics).to(torch.float32)

    return images, camtoworlds, intrinsics


class SubjectLoader_Tanks(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "trainval", "test"]
    SUBJECT_IDS = [
        "Barn",
        "Caterpillar",
        "Family",
        "Ignatius",
        "Truck",
    ]

    WIDTH, HEIGHT = 1920, 1080
    NEAR, FAR = 0.01, 6.0
    OPENGL_CAMERA = False

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        if split == "trainval":
            _images_train, _camtoworlds_train, _intrinsics_train = _load_renderings_NSVF(
                root_fp, subject_id, "train"
            )
            _images_val, _camtoworlds_val, _intrinsics_val = _load_renderings_NSVF(
                root_fp, subject_id, "val"
            )
            self.images = np.concatenate([_images_train, _images_val])
            self.camtoworlds = np.concatenate(
                [_camtoworlds_train, _camtoworlds_val]
            )
            self.K = _intrinsics_train
        else:
            self.images, self.camtoworlds, self.K = _load_renderings_NSVF(
                root_fp, subject_id, split
            )

        self.images = torch.from_numpy(self.images).to(torch.uint8)
        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        '''self.K = torch.tensor(
            [
                [self.focal, 0, self.WIDTH / 2.0],
                [0, self.focal, self.HEIGHT / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)'''
        self.images = self.images.to(device)
        self.camtoworlds = self.camtoworlds.to(device)
        self.K = self.K.to(device)
        assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)

        self.scene_bbox = torch.from_numpy(np.loadtxt(os.path.join(root_fp, subject_id, 'bbox.txt'))).float()[:6].view(2, 3)*1.2
        self.render_step_size = torch.from_numpy(np.loadtxt(os.path.join(root_fp, subject_id, 'bbox.txt'))).float()[-1].item()
        self.render_step_size = 4e-3 if self.render_step_size >= 0.15 else 1e-3

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.images.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.images.device)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.images.device)

        if rgba.shape[-1] == 4:
            pixels, alpha = torch.split(rgba, [3, 1], dim=-1)
            pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        else:
            pixels = rgba

        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays
        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(num_rays,),
                    device=self.images.device,
                )
            else:
                image_id = [index] * num_rays
            x = torch.randint(
                0, self.WIDTH, size=(num_rays,), device=self.images.device
            )
            y = torch.randint(
                0, self.HEIGHT, size=(num_rays,), device=self.images.device
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=self.images.device),
                torch.arange(self.HEIGHT, device=self.images.device),
                indexing="xy",
            )
            # x: [800, 800]
            # y: [800, 800]
            x = x.flatten()
            y = y.flatten()
            # x: [640000]
            # y: [640000]
            # num_rays = 640000

        # generate rays
        rgba = self.images[image_id, y, x] / 255.0  # (num_rays, 4)
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # camera_dirs: [num_rays, 3]
        # camera_dirs[:, -1] = value

        # [n_cams, height, width, 3]
        # [num_rays, 1, 3] * [num_rays, 3, 3] -> [num_rays, 3, 3]. sum -> [num_rays, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            try:
                rgba = torch.reshape(rgba, (num_rays, 4))
            except RuntimeError:
                rgba = torch.reshape(rgba, (num_rays, 3))
        else:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            try:
                rgba = torch.reshape(rgba, (self.HEIGHT, self.WIDTH, 4))
            except RuntimeError:
                rgba = torch.reshape(rgba, (self.HEIGHT, self.WIDTH, 3))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }
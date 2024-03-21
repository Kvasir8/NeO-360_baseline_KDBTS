import numpy as np
import os
from .kitti360_utils import *
from .ray_utils import *
from PIL import Image
from torchvision import transforms as T
import random


def read_files(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Remove any whitespace characters from the beginning or end of each line
    lines = [line.strip() for line in lines]

    # Convert the list of strings to a NumPy array
    np_array = np.array(lines)
    return np_array


class KITTI360PanopticAEDataset:
    def __init__(
        self,
        root_dir,
        split,
        sequence="2013_05_28_drive_0000_sync",
        ratio=0.5,
        img_wh=None,
        white_back=None,
        model_type=None,
        start=3353,
        train_frames=64,
        center_pose=[1360.034, 3845.649, 116.8115],
        val_list=[3365, 3375, 3385, 3395, 3405],
        eval_inference=None,
        optimize=None,
        encoder_type="resnet",
        contract=True,
        finetune_lpips=False,
    ):
        super(KITTI360PanopticAEDataset, self).__init__()
        # path and initialization

        self.val_lists = val_list
        self.img_transform = T.Compose([T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.samples_per_epoch = 6400
        self.split = split
        self.ratio = ratio
        self.cam2world_root = os.path.join(
            root_dir, "data_poses", sequence, "cam0_to_world.txt"
        )
        self.bbox_root = os.path.join(root_dir, sequence)
        self.bbx_intersection_root = os.path.join(root_dir, "bbx_intersection")
        self.sequence = sequence
        # load image_ids
        self.imgs_dir = os.path.join(root_dir, sequence)

        self.white_back = white_back

        train_ids = np.arange(start, start + train_frames)

        test_ids = np.array(val_list)

        if split == "train":
            self.image_ids = train_ids
            self.val_image_ids = test_ids
        elif split == "val":
            self.image_ids = test_ids

        self.near = 2.0
        self.far = 122.0
        self.translation = np.array(center_pose)
        self.define_transforms()
        # load intrinsics
        calib_dir = os.path.join(root_dir, "calibration")
        self.intrinsic_file = os.path.join(calib_dir, "perspective.txt")
        self.load_intrinsic(self.intrinsic_file)
        self.H = int(self.height * self.ratio)
        self.W = int(self.width * self.ratio)
        self.K_00[:2] = self.K_00[:2] * self.ratio
        self.intrinsic_00 = self.K_00[:, :-1]

        self.image_sizes = np.array([[self.H, self.W] for i in range(len(val_list))])

        # load cam2world poses
        self.cam2world_dict_00 = {}
        for line in open(self.cam2world_root, "r").readlines():
            value = list(map(float, line.strip().split(" ")))
            self.cam2world_dict_00[value[0]] = np.array(value[1:]).reshape(4, 4)

    def get_training_pairs(self):
        valid_pairs = [
            pair
            for pair in zip(self.image_ids[:-1], self.image_ids[1:])
            if all(p not in self.val_lists for p in pair)
        ]
        random_pair = np.random.choice(len(valid_pairs), size=1, replace=False)[0]
        pair = valid_pairs[random_pair]
        return pair[0], pair[1]

    def get_training_pairs_4(self):
        train_ids_filtered = np.setdiff1d(self.image_ids, self.val_lists)

        # Find sequential numbers
        valid_pairs = []
        for i in range(len(train_ids_filtered) - 3):
            if np.all(np.diff(train_ids_filtered[i : i + 4]) == 1):
                valid_pairs.append(train_ids_filtered[i : i + 4])
        random_pair = np.random.choice(len(valid_pairs), size=1, replace=False)[0]
        pair = valid_pairs[random_pair]
        return pair[:3], pair[-1]

    def define_transforms(self):
        self.transform = T.ToTensor()

    def load_intrinsic(self, intrinsic_file):
        with open(intrinsic_file) as f:
            intrinsics = f.read().splitlines()
        for line in intrinsics:
            line = line.split(" ")
            if line[0] == "P_rect_00:":
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3, 4])
                self.K_00 = K
                intrinsic_loaded = True
            elif line[0] == "S_rect_01:":
                width = int(float(line[1]))
                height = int(float(line[2]))
        assert intrinsic_loaded == True
        assert width > 0 and height > 0
        self.width, self.height = width, height

    def read_meta(self, frameId):
        frame_name = "%010d" % frameId
        image_path = os.path.join(
            self.imgs_dir, "image_00/data_rect/%s.png" % frame_name
        )
        pose = self.cam2world_dict_00[frameId]
        pose[:3, 3] = pose[:3, 3] - self.translation

        pose = openCV_to_OpenGL(pose)
        focal = self.intrinsic_00[0][0]
        c = np.array([self.intrinsic_00[0][2], self.intrinsic_00[1][2]])
        directions = get_ray_directions(self.H, self.W, focal)  # (h, w, 3)
        pose = torch.FloatTensor(pose)
        c2w = torch.FloatTensor(pose)[:3, :4]
        rays_o, viewdirs, rays_d, radii = get_rays(
            directions, c2w, output_view_dirs=True, output_radii=True
        )

        img = Image.open(image_path)
        img = img.resize((self.W, self.H), Image.LANCZOS)
        # img = self.transform(img)  # (h, w, 3)
        # img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGBA

        return (
            rays_o,
            viewdirs,
            rays_d,
            radii,
            img,
            pose,
            torch.tensor(focal, dtype=torch.float32),
            torch.tensor(c, dtype=torch.float32),
        )

    def __getitem__(self, idx):
        if self.split == "train":
            ray_batch_size = 2048
            # source_idx, target_idx = self.get_training_pairs()
            source_idx, target_idx = self.get_training_pairs_4()
            # source_idx = [source_idx]

            imgs = list()
            poses = list()
            focals = list()
            all_c = list()

            # source view data loading
            for s_idx in source_idx:
                _, _, _, _, img, c2w, f, c = self.read_meta(s_idx)
                img = Image.fromarray(np.uint8(img))
                img = T.ToTensor()(img)
                imgs.append(self.img_transform(img))
                poses.append(c2w)
                focals.append(f)
                all_c.append(c)

            imgs = torch.stack(imgs, 0)
            poses = torch.stack(poses, 0)
            focals = torch.stack(focals, 0)
            all_c = torch.stack(all_c, 0)

            # target view data loading
            rays, viewdirs, rays_d, radii, img_gt, _, _, _ = self.read_meta(target_idx)

            img_gt = Image.fromarray(np.uint8(img_gt))
            img_gt = T.ToTensor()(img_gt)
            rgb_gt = img_gt.permute(1, 2, 0).flatten(0, 1)

            rays = rays.view(-1, rays.shape[-1])
            viewdirs = viewdirs.view(-1, viewdirs.shape[-1])
            rays_d = rays_d.view(-1, rays_d.shape[-1])

            pix_inds = torch.randint(0, self.H * self.W, (ray_batch_size,))
            rgbs = rgb_gt.reshape(-1, 3)[pix_inds, ...]
            radii = radii.reshape(-1, 1)[pix_inds]
            rays = rays.reshape(-1, 3)[pix_inds]
            rays_d = rays_d.reshape(-1, 3)[pix_inds]
            view_dirs = viewdirs.reshape(-1, 3)[pix_inds]

            sample = {}
            sample["src_imgs"] = imgs
            sample["src_poses"] = poses
            sample["src_focal"] = focals
            sample["src_c"] = all_c
            sample["rays_o"] = rays
            sample["rays_d"] = rays_d
            sample["viewdirs"] = view_dirs
            sample["target"] = rgbs
            sample["radii"] = radii

            return sample

        elif (
            self.split == "val" or self.split == "test"
        ):  # create data for each image separatel
            target_idx = self.image_ids[idx]
            # source_idx = target_idx - 1

            source_idx = [target_idx - 3, target_idx - 2, target_idx - 1]

            imgs = list()
            poses = list()
            focals = list()
            all_c = list()

            # source_idx = [source_idx]
            # source view data loading
            for s_idx in source_idx:
                _, _, _, _, img, c2w, f, c = self.read_meta(s_idx)
                img = Image.fromarray(np.uint8(img))
                img = T.ToTensor()(img)
                imgs.append(self.img_transform(img))
                poses.append(c2w)
                focals.append(f)
                all_c.append(c)

            imgs = torch.stack(imgs, 0)
            poses = torch.stack(poses, 0)
            focals = torch.stack(focals, 0)
            all_c = torch.stack(all_c, 0)

            # target view data loading
            rays, viewdirs, rays_d, radii, img_gt, _, _, _ = self.read_meta(target_idx)

            img_gt = Image.fromarray(np.uint8(img_gt))
            img_gt = T.ToTensor()(img_gt)
            rgb_gt = img_gt.permute(1, 2, 0).flatten(0, 1)

            rays = rays.view(-1, rays.shape[-1])
            view_dirs = viewdirs.view(-1, viewdirs.shape[-1])
            rays_d = rays_d.view(-1, rays_d.shape[-1])

            sample = {}
            sample["src_imgs"] = imgs
            sample["src_poses"] = poses
            sample["src_focal"] = focals
            sample["src_c"] = all_c
            sample["rays_o"] = rays
            sample["rays_d"] = rays_d
            sample["viewdirs"] = view_dirs
            sample["target"] = rgb_gt
            sample["radii"] = radii
            sample["img_wh"] = np.array([self.W, self.H])
            return sample

    def __len__(self):
        if self.split == "train":
            return self.samples_per_epoch
            # train_ids = len(self.image_ids) - len(self.val_lists) - 1
            # return train_ids
        elif self.split == "val":
            return len(self.val_lists)

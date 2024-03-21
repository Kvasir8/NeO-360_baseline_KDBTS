import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from .ray_utils import *
import random
import cv2

def rot_from_origin(c2w, rotation=10):
    rot = c2w[:3, :3]
    pos = c2w[:3, -1:]
    rot_mat = get_rotation_matrix(rotation)
    pos = torch.mm(rot_mat, pos)
    rot = torch.mm(rot_mat, rot)
    c2w = torch.cat((rot, pos), -1)
    return c2w


def get_rotation_matrix(rotation):
    phi = rotation * (np.pi / 180.0)
    x = np.random.uniform(-phi, phi)
    y = np.random.uniform(-phi, phi)
    z = np.random.uniform(-phi, phi)

    rot_x = torch.Tensor(
        [[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]]
    )
    rot_y = torch.Tensor(
        [[np.cos(y), 0, -np.sin(y)], [0, 1, 0], [np.sin(y), 0, np.cos(y)]]
    )
    rot_z = torch.Tensor(
        [
            [np.cos(z), -np.sin(z), 0],
            [np.sin(z), np.cos(z), 0],
            [0, 0, 1],
        ]
    )
    rot_mat = torch.mm(rot_x, torch.mm(rot_y, rot_z))
    return rot_mat


TINY_NUMBER = 1e-6  # float32 only has 7 decimal digits precision


def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + TINY_NUMBER)
    vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(
        np.clip(np.sum(vec1_unit * vec2_unit, axis=-1), -1.0, 1.0)
    )
    return angular_dists


def batched_angular_dist_rot_matrix(R1, R2):
    """
    calculate the angular distance between two rotation matrices (batched)
    :param R1: the first rotation matrix [N, 3, 3]
    :param R2: the second rotation matrix [N, 3, 3]
    :return: angular distance in radiance [N, ]
    """
    assert (
        R1.shape[-1] == 3
        and R2.shape[-1] == 3
        and R1.shape[-2] == 3
        and R2.shape[-2] == 3
    )
    return np.arccos(
        np.clip(
            (np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1)
            / 2.0,
            a_min=-1 + TINY_NUMBER,
            a_max=1 - TINY_NUMBER,
        )
    )


def get_nearest_pose_ids(
    tar_pose,
    ref_poses,
    num_select=4,
    tar_id=-1,
    angular_dist_method="vector",
    scene_center=(0, 0, 0),
):
    """
    Args:
        tar_pose: target pose [3, 3]
        ref_poses: reference poses [N, 3, 3]
        num_select: the number of nearest views to select
    Returns: the selected indices
    """
    num_cams = len(ref_poses)
    num_select = min(num_select, num_cams - 1)
    batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0)

    if angular_dist_method == "matrix":
        dists = batched_angular_dist_rot_matrix(
            batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3]
        )
    elif angular_dist_method == "vector":
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    elif angular_dist_method == "dist":
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
    else:
        raise Exception("unknown angular distance calculation method!")

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select the target id itself

    sorted_ids = np.argsort(dists)
    selected_ids = sorted_ids[:num_select]
    # print(angular_dists[selected_ids] * 180 / np.pi)
    return selected_ids

def move_camera_pose(pose, progress):
    # control the camera move (spiral pose)
    t = progress * np.pi * 4
    radii = 0.03
    # radii = 0
    center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii
    pose[:3, 3] += pose[:3, :3] @ center
    return pose



def read_poses(pose_dir_train, img_files_train, output_boxes=False, contract=True):
    pose_file_train = os.path.join(pose_dir_train, "pose.json")
    with open(pose_file_train, "r") as read_content:
        data = json.load(read_content)

    focal = data["focal"]
    img_wh = data["img_size"]
    obj_location = np.array(data["obj_location"])
    all_c2w_train = []

    for img_file in img_files_train:        ## TODO: KITTI 360 c2w
        c2w = np.array(data["transform"][img_file.split(".")[0]])
        c2w[:3, 3] = c2w[:3, 3] - obj_location      ## a vector from the object to the camera in world space
        all_c2w_train.append(convert_pose_PD_to_NeRF(c2w))      

    all_c2w_train = np.array(all_c2w_train)
    pose_scale_factor = 1.0 / np.max(np.abs(all_c2w_train[:, :3, 3]))   ## why do we need "pose_scale_factor"? : to scale the pose to the same scale as the 3D model

    all_c2w_train[:, :3, 3] *= pose_scale_factor

    all_c2w_val = all_c2w_train[100:]
    all_c2w_train = all_c2w_train[:100]
    # Get bounding boxes for object MLP training only
    use_pred_box = False
    if output_boxes:
        all_boxes = []
        all_translations = []
        all_rotations = []
        if use_pred_box:
            box_file = os.path.join(
                pose_dir_train, "box_predicted_procrustes_testprior.json"
            )
            with open(box_file, "r") as read_content:
                data = json.load(read_content)
            for k, v in data["bbox_dimensions"].items():
                bbox = np.array(v)
                all_boxes.append(bbox * pose_scale_factor)
                # New scene 200 uncomment here
                all_rotations.append(data["obj_rotations"][k])
                translation = (
                    np.array(data["obj_translations"][k])
                ) * pose_scale_factor
                all_translations.append(translation)
        else:
            for k, v in data["bbox_dimensions"].items():
                bbox = np.array(v)
                all_boxes.append(bbox * pose_scale_factor)
                # New scene 200 uncomment here
                all_rotations.append(data["obj_rotations"][k])
                translation = (
                    np.array(data["obj_translations"][k]) - obj_location
                ) * pose_scale_factor
                all_translations.append(translation)
        # Old scenes uncomment here
        RTs = {"R": all_rotations, "T": all_translations, "s": all_boxes}
        return all_c2w_train, all_c2w_val, focal, img_wh, RTs, pose_scale_factor    ### len(RTs['R'][0][0]) == 3
    else:
        # return all_c2w_train, all_c2w_val, focal, img_wh, pose_scale_factor
        RTs = None
        return all_c2w_train, all_c2w_val, focal, img_wh, RTs , pose_scale_factor


def read_poses_val(pose_dir_train, img_files_train, pose_scale_factor):
    pose_file_train = os.path.join(pose_dir_train, "pose.json")
    with open(pose_file_train, "r") as read_content:
        data = json.load(read_content)

    obj_location = np.array(data["obj_location"])
    all_c2w_train = []

    for img_file in img_files_train:
        c2w = np.array(data["transform"][img_file.split(".")[0]])
        c2w[:3, 3] = c2w[:3, 3] - obj_location
        all_c2w_train.append(convert_pose_PD_to_NeRF(c2w))

    all_c2w_train = np.array(all_c2w_train)
    all_c2w_train[:, :3, 3] *= pose_scale_factor

    return all_c2w_train


class NeRDS360_AE_custom(Dataset):
    def __init__(
        self,
        root_dir = "",
        split="train",
        ratio=0.5,
        img_wh= (640, 192),      ## default: (640, 480) for NeO360
        white_back=False,
        model_type="kitti360",
        # eval_inference=None,
        # optimize=None,
        # encoder_type="resnet",
        contract=True,
        finetune_lpips=False,
        ray_batch_size=2048,
    ):
        self.ratio = ratio
        self.split = split
        self.img_wh = img_wh
        self.define_transforms()
        self.white_back = white_back
        self.base_dir = root_dir
        # self.ids = np.sort([f.name for f in os.scandir(self.base_dir)])
        self.eval_inference = eval_inference
        # self.optimize = optimize
        self.encoder_type = encoder_type
        self.contract = contract
        self.finetune_lpips = finetune_lpips
        self.ray_batch_size = ray_batch_size

        # for multi scene training
        if self.encoder_type == "resnet":
            self.img_transform = T.Compose(
                [T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        else:
            # for custom CNN MVS nerf style
            self.img_transform = T.Compose(
                [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            )
        self.samples_per_epoch = 6400

        if self.eval_inference is not None:
            # num = 3
            num = 99
            # num = 40
            self.image_sizes = np.array([[h, w] for i in range(num)])
        else:
            self.image_sizes = np.array([[h, w] for i in range(1)])

        self.model_type = model_type
        self.near, self.far = 2.0, 122.0     ## default: 3.0, 80.0 for BTS

        self.width, self.height = img_wh
        # self.H, self.W = int(self.height * self.ratio), int(self.width * self.ratio)
        self.H, self.W = int(self.height), int(self.width)


    def read_data(self, **kwargs):
        focal = kwargs["focal"]
        c2w = kwargs["c2w"]
        # c = np.array([self.W / 2.0, self.H / 2.0])    ## 640 192
        c = np.array([self.intrinsic_00[0][2], self.intrinsic_00[1][2]])
        pose = torch.FloatTensor(c2w)
        c2w = torch.FloatTensor(c2w)[:3, :4]        ## data redundancy
        directions = get_ray_directions(self.H, self.W, focal[0])  # (h, w, 3)
        rays_o, view_dirs, rays_d, radii = get_rays(
            directions, c2w, output_view_dirs=True, output_radii=True
        )

        img = img.resize((self.W, self.H), Image.LANCZOS)

        return (
            rays_o,
            view_dirs,
            rays_d,
            radii,
            img,
            pose,
            torch.tensor(focal, dtype=torch.float32),
            torch.tensor(c, dtype=torch.float32),
        )

    def get_training_pairs_4(self): ####
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

    def __getitem__(self, idx: int):
        if self.split == "train":  # use data in the buffers
            ray_batch_size = self.ray_batch_size
            # train_idx = random.randint(0, len(self.ids) - 1)
            source_idx, target_idx = self.get_training_pairs_4()

            imgs = list()
            poses = list()
            focals = list()
            all_c = list()

            # source view data loading
            for s_idx in source_idx:
                _, _, _, _, img, c2w, f, c = self.read_data(s_idx)
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
            rays, viewdirs, rays_d, radii, img_gt, _, _, _ = self.read_data(target_idx)

            img_gt = Image.fromarray(np.uint8(img_gt))
            img_gt = T.ToTensor()(img_gt)
            rgb_gt = img_gt.permute(1, 2, 0).flatten(0, 1)

            rays = rays.view(-1, rays.shape[-1])
            viewdirs = viewdirs.view(-1, viewdirs.shape[-1])
            rays_d = rays_d.view(-1, rays_d.shape[-1])
        
            pix_inds = torch.randint(0, self.H * self.W, (ray_batch_size,))
            rgbs = rgbs.reshape(-1, 3)[pix_inds, ...]
            radii = radii.reshape(-1, 1)[pix_inds]
            rays = rays.reshape(-1, 3)[pix_inds]
            rays_d = rays_d.reshape(-1, 3)[pix_inds]
            view_dirs = view_dirs.reshape(-1, 3)[pix_inds]

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

        elif self.split == "val" or self.split == "test":
            # create data for each image separatel
            target_idx = self.image_ids[idx]
            # source_idx = target_idx - 1

            source_idx = [target_idx - 3, target_idx - 2, target_idx - 1]

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
            rays, viewdirs, rays_d, radii, img_gt, _, _, _ = self.read_data(target_idx)

            img_gt = Image.fromarray(np.uint8(img_gt))
            img_gt = T.ToTensor()(img_gt)
            rgb_gt = img_gt.permute(1, 2, 0).flatten(0, 1)
            
            rays = rays.view(-1, rays.shape[-1])
            rays_d = rays_d.view(-1, rays_d.shape[-1])
            view_dirs = viewdirs.view(-1, viewdirs.shape[-1])

            if self.model_type == "Vanilla":
                sample = {
                    "src_imgs": imgs,
                    "rays": rays,
                    "rgbs": rgbs,
                }
            elif self.model_type == "kitti360":
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
            else: raise NotImplementedError(f"Model type not implemented{self.model_type}")

            return sample

        else: raise NotImplementedError(f"Split not implemented{self.split}")

    def __len__(self):
        if self.split == "train":
            return self.samples_per_epoch
        elif self.split == "val":
            # return len(self.val_lists)
            return pass
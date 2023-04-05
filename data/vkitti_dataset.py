import glob
import os.path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils import data


def get_surface_normals(depth, k=None):
    k = np.array([[725.0087, 0, 620.5], [0, 725.0087, 187], [0, 0, 1]]) if k is None else k
    height, width = depth.shape

    def normalization(dat):
        mo_chang = np.sqrt(
            np.multiply(dat[:, :, 0], dat[:, :, 0])
            + np.multiply(dat[:, :, 1], dat[:, :, 1])
            + np.multiply(dat[:, :, 2], dat[:, :, 2])
        )
        mo_chang = np.dstack((mo_chang, mo_chang, mo_chang))
        return dat / mo_chang

    x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x = x.reshape([-1])
    y = y.reshape([-1])
    xyz = np.vstack((x, y, np.ones_like(x)))
    pts_3d = np.dot(np.linalg.inv(k), xyz * depth.reshape([-1]))
    pts_3d_world = pts_3d.reshape((3, height, width))
    f = (
            pts_3d_world[:, 1: height - 1, 2:width]
            - pts_3d_world[:, 1: height - 1, 1: width - 1]
    )
    t = (
            pts_3d_world[:, 2:height, 1: width - 1]
            - pts_3d_world[:, 1: height - 1, 1: width - 1]
    )
    normal_map = np.cross(f, t, axisa=0, axisb=0)
    normal_map = normalization(normal_map)

    return normal_map.astype(np.float32)


class VKITTIDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.image_list = None
        self.opt = opt
        self.batch_size = opt.batch_size
        self.use_surface_normal = opt.use_sn
        self.root = opt.data_root
        self.num_labels = 2
        self.use_size = (opt.resize_width, opt.resize_height)
        if opt.phase == "train":
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'training', 'rgb', '*.jpg')))
        elif opt.phase == "val":
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'validation', 'rgb', '*.jpg')))
        else:
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'testing', 'rgb', '*.jpg')))

    def __getitem__(self, index):
        use_dir = "/".join(self.image_list[index].split('\\')[:-2])
        name = self.image_list[index].split('\\')[-1][4:-4]

        rgb_image = cv2.imread(os.path.join(use_dir, 'rgb', "rgb_" + name + ".jpg"))
        depth_image = cv2.imread(os.path.join(use_dir, 'depth', "depth_" + name + ".png"), cv2.IMREAD_ANYDEPTH)
        depth_in_meters = 655.35 / (2 ** 16 - 1) * depth_image.astype(np.float32)  # convert from cm to m
        orig_height, orig_width, _ = rgb_image.shape
        if self.opt.phase == 'test' and self.opt.no_label:
            # Since we have no gt label, we generate pseudo gt labels
            label = np.zeros((orig_height, orig_width), dtype=np.uint8)
        else:
            label_image = cv2.imread(os.path.join(use_dir, 'gt2', "classgt_" + name + ".png"))
            label = np.zeros((orig_height, orig_width), dtype=np.uint8)
            label[(label_image == [100, 60, 100]).all(axis=2)] = 1

        # resize image to enable sizes divide 32
        rgb_image = cv2.resize(rgb_image, self.use_size)
        label = cv2.resize(label, self.use_size, interpolation=cv2.INTER_NEAREST)

        # another_image will be normal when using SNE, otherwise will be depth
        if self.use_surface_normal:
            another_image = get_surface_normals(depth_in_meters)
            another_image = cv2.resize(another_image, self.use_size)
        else:
            inv_depth = depth_in_meters
            another_image = inv_depth / np.max(inv_depth)
            another_image = cv2.resize(another_image, self.use_size)
            another_image = another_image[:, :, np.newaxis]

        label[label > 0] = 1
        rgb_image = rgb_image.astype(np.float32) / 255

        rgb_image = transforms.ToTensor()(rgb_image)
        another_image = transforms.ToTensor()(another_image)

        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)

        # return a dictionary containing useful information
        # input rgb images, another images and labels for training;
        # 'path': image name for saving predictions
        # 'orig_size': original image size for evaluating and saving predictions
        return {'rgb_image': rgb_image, 'another_image': another_image, 'label': label,
                'path': name, 'orig_size': (orig_width, orig_height)}

    def __len__(self):
        return len(self.image_list)

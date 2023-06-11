import glob
import os.path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils import data

from util.util import get_surface_normals


class VKITTIDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.image_list = None
        self.opt = opt
        self.num_labels = 2
        self.use_size = (opt.resize_width, opt.resize_height)
        if opt.phase == "train":
            self.image_list = sorted(glob.glob(os.path.join(self.opt.data_root, 'training', 'rgb', '*.jpg')))
        elif opt.phase == "val":
            self.image_list = sorted(glob.glob(os.path.join(self.opt.data_root, 'validation', 'rgb', '*.jpg')))
        else:
            self.image_list = sorted(glob.glob(os.path.join(self.opt.data_root, 'testing', 'rgb', '*.jpg')))

    def __getitem__(self, index):
        use_dir = "/".join(self.image_list[index].split('\\')[:-2])
        name = self.image_list[index].split('\\')[-1][4:-4]

        rgb_image = cv2.cvtColor(cv2.imread(os.path.join(use_dir, 'rgb', "rgb_" + name + ".jpg")), cv2.COLOR_BGR2RGB)
        depth_image = cv2.imread(os.path.join(use_dir, 'depth', "depth_" + name + ".png"), cv2.IMREAD_ANYDEPTH)
        orig_height, orig_width, _ = rgb_image.shape
        if self.opt.phase == 'test' and self.opt.no_label:
            label = np.zeros((orig_height, orig_width), dtype=np.uint8)
        else:
            label_image = cv2.imread(os.path.join(use_dir, 'gt2', "classgt_" + name + ".png"))
            label = np.zeros((orig_height, orig_width), dtype=np.uint8)
            label[(label_image == [100, 60, 100]).all(axis=2)] = 1

        rgb_image = cv2.resize(rgb_image, self.use_size)
        label = cv2.resize(label, self.use_size, interpolation=cv2.INTER_NEAREST)

        if self.opt.use_sne:
            k = np.array([[725.0087, 0, 620.5], [0, 725.0087, 187], [0, 0, 1]])
            another_image = get_surface_normals(depth_image.astype(np.float32) / 100, k)
            another_image = cv2.resize(another_image, self.use_size)
        else:
            another_image = depth_image.astype(np.float32) / 65535
            another_image = cv2.resize(another_image, self.use_size)
            another_image = another_image[:, :, np.newaxis]

        label[label > 0] = 1
        rgb_image = rgb_image.astype(np.float32) / 255

        rgb_image = transforms.ToTensor()(rgb_image)
        another_image = transforms.ToTensor()(another_image)

        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)

        return {'rgb_image': rgb_image, 'another_image': another_image, 'label': label,
                'path': name, 'orig_size': (orig_width, orig_height)}

    def __len__(self):
        return len(self.image_list)

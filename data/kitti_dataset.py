import glob
import os.path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils import data

from util.util import get_surface_normals


class KITTIDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.image_list = None
        self.opt = opt
        self.batch_size = opt.batch_size
        self.use_surface_normal = opt.use_sne
        self.root = opt.data_root
        self.num_labels = 2
        self.use_size = (opt.resize_width, opt.resize_height)
        if self.use_surface_normal:
            pass
        if opt.phase == "train":
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'training', 'image_2', '*.png')))
        elif opt.phase == "val":
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'validation', 'image_2', '*.png')))
        else:
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'testing', 'image_2', '*.png')))

    def __getitem__(self, index):
        use_dir = "/".join(self.image_list[index].split('\\')[:-2])
        name = self.image_list[index].split('\\')[-1][:-4]

        rgb_image = cv2.imread(os.path.join(use_dir, 'image_2', name + ".png"))
        depth_image = cv2.imread(os.path.join(use_dir, 'depth2', name + ".png"), cv2.IMREAD_ANYDEPTH)
        orig_height, orig_width, _ = rgb_image.shape
        if self.opt.phase == 'test' and self.opt.no_label:
            # Since we have no gt label, we generate pseudo gt labels
            label = np.zeros((orig_height, orig_width), dtype=np.uint8)
        else:
            label_image = cv2.imread(os.path.join(use_dir, 'gt_image_2', name + ".png"))
            label = np.zeros((orig_height, orig_width), dtype=np.uint8)
            label[(label_image == [255, 0, 255]).all(axis=2)] = 1

        # resize image to enable sizes divide 32
        rgb_image = cv2.resize(rgb_image, self.use_size)
        label = cv2.resize(label, self.use_size, interpolation=cv2.INTER_NEAREST)

        # another_image will be normal when using SNE, otherwise will be depth
        if self.use_surface_normal:
            k = [[7.215377e+02, 0.000000e+00, 6.095593e+02],
                 [0.000000e+00, 7.215377e+02, 1.728540e+02],
                 [0.000000e+00, 0.000000e+00, 1.000000e+00]]
            another_image = get_surface_normals(depth_image.astype(np.float32) / 1000, k)
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

        # return a dictionary containing useful information
        # input rgb images, another images and labels for training;
        # 'path': image name for saving predictions
        # 'orig_size': original image size for evaluating and saving predictions
        return {'rgb_image': rgb_image, 'another_image': another_image, 'label': label,
                'path': name, 'orig_size': (orig_width, orig_height)}

    def __len__(self):
        return len(self.image_list)

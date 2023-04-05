import torch

from . import networks
from .base_model import BaseModel, set_requires_grad
from .stixelnet import StixelNet


class StixelNetModel(BaseModel):
    def __init__(self, opt, dataset):
        super().__init__(opt)
        self.is_train = opt.is_train

        self.loss_names = ['stixel_loss']
        self.model_name = 'StixelNet'
        self.net = StixelNet()
        self.output = None
        self.image_orig_size = None
        self.label = None
        self.rgb_image = None

    def set_input(self, data):
        self.rgb_image = data['rgb_image'].to(self.device)
        self.label = data['label'].to(self.device)
        self.image_names = data['path']
        self.image_orig_size = data['orig_size']

    def forward(self):
        self.output = self.net(self.rgb_image)

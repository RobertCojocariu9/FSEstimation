import torch

from . import networks
from .base_model import BaseModel, set_requires_grad


class FSNetModel(BaseModel):
    def __init__(self, opt, dataset):
        super().__init__(opt)
        self.is_train = opt.is_train

        self.loss_names = ['cross_entropy']
        self.visual_names = ['rgb_image', 'another_image', 'label', 'output']
        self.model_name = 'FSNet'
        self.net = networks.init_net(dataset.num_labels, use_sne=opt.use_sn,
                                     init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        self.criterion_cross_entropy = torch.nn.CrossEntropyLoss().to(self.device)

        if self.is_train:
            self.optimizers = []
            self.optimizer_semseg = torch.optim.SGD(self.net.parameters(), lr=opt.lr, momentum=opt.momentum,
                                                    weight_decay=opt.weight_decay)
            # self.optimizer_semseg = torch.optim.Adam(self.netFSNet.parameters(), lr=opt.lr,
            #                                          weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer_semseg)
            set_requires_grad(self.net, True)
        self.loss_cross_entropy = None
        self.output = None
        self.image_orig_size = None
        self.label = None
        self.another_image = None
        self.rgb_image = None

    def set_input(self, data):
        self.rgb_image = data['rgb_image'].to(self.device)
        self.another_image = data['another_image'].to(self.device)
        self.label = data['label'].to(self.device)
        self.image_names = data['path']
        self.image_orig_size = data['orig_size']

    def forward(self):
        self.output = self.net(self.rgb_image, self.another_image)

    def get_loss(self):
        self.loss_cross_entropy = self.criterion_cross_entropy(self.output, self.label)

    def backward(self):
        self.loss_cross_entropy.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_semseg.zero_grad()
        self.get_loss()
        self.backward()
        self.optimizer_semseg.step()

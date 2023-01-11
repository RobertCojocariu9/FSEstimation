import torch

from . import networks
from .base_model import BaseModel


class FSNetModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.loss_segmentation = None
        self.output = None
        self.image_orig_size = None
        self.label = None
        self.another_image = None
        self.rgb_image = None
        self.optimizer_semseg = None
        self.criterion_segmentation = None
        self.netFSNet = None

    def name(self):
        return 'FSNet'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def initialize(self, opt, dataset):
        BaseModel.initialize(self, opt, dataset)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['segmentation']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['rgb_image', 'another_image', 'label', 'output']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and
        # base_model.load_networks
        self.model_names = ['FSNet']

        # load/define networks
        self.netFSNet = networks.define_fsnet(dataset.num_labels, use_sne=opt.use_surface_normal,
                                              init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        # define loss functions
        self.criterion_segmentation = torch.nn.CrossEntropyLoss().to(self.device)

        if self.isTrain:
            # initialize optimizers
            self.optimizers = []
            self.optimizer_semseg = torch.optim.SGD(self.netFSNet.parameters(), lr=opt.lr, momentum=opt.momentum,
                                                    weight_decay=opt.weight_decay)
            # self.optimizer_semseg = torch.optim.Adam(self.netRoadSeg.parameters(), lr=opt.lr,
            # weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer_semseg)
            self.set_requires_grad(self.netFSNet, True)

    def set_input(self, input):
        self.rgb_image = input['rgb_image'].to(self.device)
        self.another_image = input['another_image'].to(self.device)
        self.label = input['label'].to(self.device)
        self.image_names = input['path']
        self.image_orig_size = input['orig_size']

    def forward(self):
        self.output = self.netFSNet(self.rgb_image, self.another_image)

    def get_loss(self):
        self.loss_segmentation = self.criterion_segmentation(self.output, self.label)

    def backward(self):
        self.loss_segmentation.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_semseg.zero_grad()
        self.get_loss()
        self.backward()
        self.optimizer_semseg.step()

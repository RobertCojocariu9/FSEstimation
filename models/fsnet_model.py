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
        self.net = networks.init_net(dataset.num_labels, use_sne=opt.use_sne,
                                     init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        self.criterion_cross_entropy = torch.nn.CrossEntropyLoss().to(self.device)

        if self.is_train:
            self.optimizers = []
            self.optimizer_semseg = torch.optim.SGD(self.net.parameters(), lr=opt.lr, momentum=opt.momentum,
                                                    weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer_semseg)
            set_requires_grad(self.net, True)
        self.loss_cross_entropy = None
        self.output = None
        self.image_orig_size = None
        self.label = None
        self.another_image = None
        self.rgb_image = None

    def set_input(self, data):
        """
        Sets the input data for the model
        :param data: Dict containing the inputs (given by the Dataset)
        """
        self.rgb_image = data['rgb_image'].to(self.device)
        self.another_image = data['another_image'].to(self.device)
        self.label = data['label'].to(self.device)
        self.image_names = data['path']
        self.image_orig_size = data['orig_size']

    def forward(self):
        """
        Performs a forward pass
        """
        self.output = self.net(self.rgb_image, self.another_image)

    def get_loss(self):
        """
        Computes the loss based on the ground truth and the prediction
        """
        self.loss_cross_entropy = self.criterion_cross_entropy(self.output, self.label)

    def backward(self):
        """
        Performs the backpropagation of gradients
        """
        self.loss_cross_entropy.backward()

    def optimize_parameters(self):
        """
        Performs a full run of the data through the network (forward pass, loss computation and backpropagation)
        """
        self.forward()
        self.optimizer_semseg.zero_grad()
        self.get_loss()
        self.backward()
        self.optimizer_semseg.step()

import os
import torch
from collections import OrderedDict
from . import networks


def set_requires_grad(nets, requires_grad=False):
    """
    Sets 'requires_grad' attribute of the given network(s)
    :param nets: The network or list of networks
    :param requires_grad: Boolean, whether to enable or disable the attributed
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class BaseModel:
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_root, opt.name)
        self.loss_names = []
        self.model_name = ""
        self.visual_names = []
        self.image_names = []
        self.image_orig_size = []
        self.optimizers = None
        self.schedulers = None

    def setup(self):
        """
        Performs the setup of the network
        """
        if self.is_train:  # initialize lr schedulers
            self.schedulers = [networks.get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]

        if not self.is_train or self.opt.continue_train:  # load network if already exists
            self.load_network(self.opt.epoch)
        self.print_networks()

    def forward(self):
        pass

    def eval(self):
        net = getattr(self, 'net')
        net.eval()

    def train(self):
        net = getattr(self, 'net')
        net.train()

    def optimize_parameters(self):
        pass

    def update_learning_rate(self, valid_loss=None):
        """
        Updates the learning rate of the model's optimizers
        :param valid_loss: Validation loss value, required for some schedulers
        """
        for scheduler in self.schedulers:
            if valid_loss is not None:
                scheduler.step(valid_loss)
            else:
                scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('Learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """
        Wraps the visuals in a dict {name: visual}
        :return: Resulted dict
        """
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """
        Wraps the losses in a dict {name: loss_value}
        :return: Resulted dict
        """
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_network(self, epoch):
        """
        Saves to disk the network, under the format '[epoch]_net_[model_name].pth'
        :param epoch: Name of the network's epoch
        """
        save_filename = '%s_net_%s.pth' % (epoch, self.model_name)
        save_path = os.path.join(self.save_dir, save_filename)
        net = getattr(self, 'net')

        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(net.module.cpu().state_dict(), save_path)
            net.cuda(self.gpu_ids[0])
        else:
            torch.save(net.cpu().state_dict(), save_path)

    def load_network(self, epoch):
        """
        Loads from disk the specified network, saved under the format '[epoch]_net_[model_name].pth'
        :param epoch: Name of the network's epoch
        """
        load_filename = '%s_net_%s.pth' % (epoch, self.model_name)
        load_path = os.path.join(self.save_dir, load_filename)
        net = getattr(self, 'net')
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('Loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        net.load_state_dict(state_dict)

    def print_networks(self):
        """
        Prints the layers of the network
        """
        print('------------- Network initialized -------------')
        net = getattr(self, 'net')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('[Network %s] Total number of parameters : %.3f M' % (self.model_name, num_params / 1e6))
        print('------------------------------------------------')

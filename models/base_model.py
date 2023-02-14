import os
import torch
from collections import OrderedDict
from . import networks


def set_requires_grad(nets, requires_grad=False):
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
        if self.is_train:
            self.schedulers = [networks.get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]

        if not self.is_train or self.opt.continue_train:
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

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('Learning rate = %.7f' % lr)

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_network(self, epoch):
        save_filename = '%s_net_%s.pth' % (epoch, self.model_name)
        save_path = os.path.join(self.save_dir, save_filename)
        net = getattr(self, 'net')

        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(net.module.cpu().state_dict(), save_path)
            net.cuda(self.gpu_ids[0])
        else:
            torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_network(self, epoch):
        load_filename = '%s_net_%s.pth' % (epoch, self.model_name)
        load_path = os.path.join(self.save_dir, load_filename)
        net = getattr(self, 'net')
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('Loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        for key in list(state_dict.keys()):
            self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        net.load_state_dict(state_dict)

    def print_networks(self):
        print('------------- Network initialized -------------')
        net = getattr(self, 'net')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('[Network %s] Total number of parameters : %.3f M' % (self.model_name, num_params / 1e6))
        print('------------------------------------------------')

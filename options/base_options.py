import os

from util import util


class BaseOptions:
    data_root = "datasets\\kitti"  # dir of dataset
    batch_size = 1
    resize_width = 1248
    resize_height = 384
    gpu_ids = [0]
    name = "kitti"  # folder name for storing models and samples
    use_sn = False  # use surface norma module
    dataset = "kitti"  # dataset name
    model = "fsnet"  # model name
    epoch = "best"  # name of epoch, if starting from an existing checkpoint
    threads = 8
    checkpoints_root = "./checkpoints"  # dir of checkpoints
    serial_batches = False  # take images in order or randomly when creating batches
    init_type = "kaiming"  # weight initialization (normal, kaiming, xavier, orthogonal)
    init_gain = 0.02  # scale factor for normal, xavier and orthogonal
    seed = 0  # seed for random
    is_train = True

    def print_options(self, opt):
        message = ''
        message += '----------------- Options -----------------\n'
        fields = [attr for attr in dir(self) if not callable(getattr(self, attr))]
        for field in fields:
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(field), str(getattr(self, field)), comment)
        message += '-----------------   End   -----------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

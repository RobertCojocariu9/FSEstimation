import os

from util import util


class BaseOptions:
    data_root = "datasets\\vkitti"  # dir of dataset
    batch_size = 16
    resize_width = 480
    resize_height = 160
    gpu_ids = [0]
    name = "vkitti"  # folder name for storing models and samples
    use_sn = True  # use surface normals instead of depth
    dataset = "vkitti"  # dataset name
    model = "fsnet"  # model name
    epoch = "gt2res"  # name of epoch, if starting from an existing checkpoint
    threads = 8
    checkpoints_root = "checkpoints"  # dir of checkpoints
    serial_batches = False  # take images in order or randomly when creating batches
    init_type = "kaiming"  # weight initialization (normal, kaiming, xavier, orthogonal)
    init_gain = 0.02  # scale factor for normal, xavier and orthogonal
    seed = 0  # seed for random
    is_train = True

    def __init__(self):
        self.print_options()

    def print_options(self):
        message = ''
        message += '----------------- Options -----------------\n'
        fields = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        for field in fields:
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(field), str(getattr(self, field)), comment)
        message += '-----------------   End   -----------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(self.checkpoints_root, self.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

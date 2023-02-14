from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    print_freq = 10  # print frequency of results
    continue_train = False  # if set, an existing model will be loaded
    phase = "train"  # train or val
    epoch_count = 50  # total amount of epochs
    lr = 0.001
    momentum = 0.9
    weight_decay = 0.0005
    lr_scheduler = "lstep"  # lstep, step, plateau, cosine
    lr_decay_iter = 5000  # multiply by gamma every lr_decay_iter iterations (step)
    lr_decay_epoch = 25  # multiply by gamma every lr_decay_epoch epochs (lstep)
    lr_gamma = 0.9  # gamma factor
    is_train = True

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    print_freq = 10  # print frequency of results
    continue_train = False  # if set, an existing model will be loaded
    phase = "train"  # train or val
    epoch_count = 25  # total amount of epochs
    lr = 0.01
    momentum = 0.9
    weight_decay = 0.0005
    lr_scheduler = "lstep"  # lstep, step, plateau, cosine
    lr_decay_iter = 7  # multiply by gamma every lr_decay_iter iterations (step)
    lr_decay_epoch = 10  # multiply by gamma every lr_decay_epoch epochs (lstep)
    lr_gamma = 0.1  # gamma factor
    is_train = True

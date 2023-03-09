from .base_options import BaseOptions


class TestOptions(BaseOptions):
    results_root = "results"  # results dir
    prob_map = True  # output probability map or binary prediction
    no_label = False  # set if gt labels not present
    is_train = False
    phase = "test"

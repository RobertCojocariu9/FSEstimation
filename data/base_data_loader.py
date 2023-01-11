class BaseDataLoader:
    def __init__(self):
        self.opt = None

    def initialize(self, opt):
        self.opt = opt

    def load_data(self):
        return None

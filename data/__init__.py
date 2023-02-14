import torch.utils.data

from data.kitti_dataset import KITTIDataset


def create_dataset(opt):
    if opt.dataset == "kitti":
        dataset = KITTIDataset(opt)
    else:
        print("Dataset [%s] does not exist" % opt.dataset)
        exit(1)
    print("Dataset [%s] was created" % dataset.__class__.__name__)
    return dataset


class DataLoaderWrapper:
    def __init__(self, opt):
        self.dataset = create_dataset(opt)
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=0)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.data_loader):
            yield data

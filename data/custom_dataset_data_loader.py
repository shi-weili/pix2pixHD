import torch.utils.data
from data.base_data_loader import BaseDataLoader

def CreateDataset(opt):
    dataset = None
    if opt.dataset == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()
    elif opt.dataset == 'simple_grid':
        from data.simple_grid_dataset import SimpleGridDataset
        dataset = SimpleGridDataset()
    elif opt.dataset == 'seamless_grid':
        from data.seamless_grid_dataset import SeamlessGridDataset
        dataset = SeamlessGridDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

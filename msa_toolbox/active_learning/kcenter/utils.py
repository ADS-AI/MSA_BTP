import torch
from torch.utils.data import DataLoader, Dataset


class WhiteBoxDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=2, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None):
        super(WhiteBoxDataLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context)

    def __iter__(self):
        return WhiteBoxDataLoaderIterator(self)

class WhiteBoxDataLoaderIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.data_iterator = iter(dataloader.dataset)
        self.batch_size = dataloader.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        images = []
        indices = []
        for _ in range(self.batch_size):
            try:
                item = next(self.data_iterator)
                images.append(item[0])
                indices.append(item[2])
            except StopIteration:
                break
        if not images:
            raise StopIteration
        images = torch.stack(images)
        indices = torch.tensor(indices)
        return images, None, indices

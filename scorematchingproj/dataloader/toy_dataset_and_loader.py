import numpy as np
import torch
from torch.utils import data
from torch.utils.data import dataset

from scorematchingproj.dataloader.gen_toy_samples import *


def _make_torch_dataset_to_infinite_generator(dataset, is_random=False):
    num_elt = len(dataset)
    while True:
        # print(f"Init dataset {dataset.name}, and its len = {num_elt}")
        indices = torch.randperm(num_elt).tolist() if is_random else torch.arange(num_elt).tolist()
        for idx in indices:
            yield dataset[idx]


class ToyDataset(dataset.Dataset):
    def __init__(self, set_type="2spirals", n_samples=1000000):
        super().__init__()
        if set_type not in ["2spirals", "8gaussians", "checkerboard", "rings"]:
            raise ValueError("`set_type` should be one of [`2spirals`, `8gaussians`, `checkerboard`, `rings`]")

        print(f"generating {n_samples} of data")
        if set_type == "8gaussians":
            data = gen_8gaussians(n_samples)
        elif set_type == "2spirals":
            data = gen_2spirals(n_samples)
        elif set_type == "checkerboard":
            data = gen_checkerboard(n_samples)
        else:  # set_type=='rings'
            data = gen_rings(n_samples)
        print(f"done...")

        self._data = data  # (n_samples, 2)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def get_data(self):
        return self._data


class ToyIterDataset(dataset.IterableDataset):
    def __init__(self, set_type="2spirals", n_samples=1000000):
        self._dataset = ToyDataset(set_type, n_samples)

    def __iter__(self):
        dataset_generator = _make_torch_dataset_to_infinite_generator(self._dataset, is_random=True)
        while True:
            yield next(dataset_generator)

    def get_data(self):
        return self._dataset.get_data()


class ToyDataLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=100, num_workers=0, loader_kwargs={}):
        if not (isinstance(dataset, ToyIterDataset) or isinstance(dataset, ToyDataset)):
            raise ValueError("dataset only be type `ToyIterDataset` or `ToyDataset`")
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, **loader_kwargs)
        self._dataset = dataset

    def get_data(self):
        return self._dataset.get_data()


if __name__ == "__main__":
    # Comment out `from scorematchingproj.dataloader.gen_toy_samples import *` first
    from gen_toy_samples import *

    toy_iter_dataset = ToyIterDataset()
    for _, data in zip(range(10), toy_iter_dataset):
        print(data)

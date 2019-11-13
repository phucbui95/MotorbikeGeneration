from torch.utils.data import SubsetRandomSampler, Dataset, DataLoader
import os
import torch
import argparse
from dataset import MotorbikeWithLabelsDataset, get_transforms
import torch
import numpy as np
from functools import partial

def sample_latent_vector(class_distributions, latent_size, batch_size, device):
    """Util function to sample latent vectors from specified distribution"""
    noise = torch.randn(batch_size, latent_size, 1, 1, device=device)
    n_classes = len(class_distributions)
    aux_labels = np.random.choice(n_classes, batch_size, p=class_distributions)
    aux_labels_ohe = np.eye(n_classes)[aux_labels]
    aux_labels_ohe = torch.from_numpy(
        aux_labels_ohe[:, :, np.newaxis, np.newaxis])
    aux_labels_ohe = aux_labels_ohe.float().to(device, non_blocking=True)
    aux_labels = torch.from_numpy(aux_labels).to(device)
    return noise, aux_labels, aux_labels_ohe


def test_sample_latent_vector():
    class_dist = [0.1, 0.3, 0.6]
    latent_size = 100
    batch_size = 4
    device = torch.device('cpu')
    sample_label_lst = []
    for i in range(10000):
        _, aux_labels, _ = sample_latent_vector(class_dist, latent_size,
                                                batch_size, device)
        sample_label_lst.append(aux_labels)

    sample_labels = np.concatenate(sample_label_lst, axis=0)
    counter = {}
    for s in sample_labels:
        if s in counter:
            counter[s] += 1
        else:
            counter[s] = 1

    dist = np.zeros(len(class_dist))
    for i in range(len(class_dist)):
        dist[i] = counter[i] / len(sample_labels)
    print(dist)

class MotorbikeDataloader:
    def __init__(self, opt, dataset):
        if opt.shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size,
            shuffle=(train_sampler is None),
            num_workers=opt.workers,
            pin_memory=True,
            sampler=train_sampler)
        self.opt = opt
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def get_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()
        return batch.to(self.get_device())

    def get_latent_sample_fnc(self, latent_size=None, batch_size=None, device=None):
        fnc = partial(sample_latent_vector, class_distributions=self.dataset.get_class_distributions(),
                      latent_size=self.opt.latent_size if latent_size is None else latent_size,
                      batch_size=self.opt.batch_size if batch_size is None else batch_size,
                      device=self.get_device() if device is None else device)
        return fnc

    def latent_sample(self):
        class_dist = self.dataset.get_class_distributions()
        return sample_latent_vector(class_dist, self.opt.latent_size, self.opt.batch_size, self.get_device())

if __name__ == '__main__':

    args = argparse.ArgumentParser(description="Dataloader testing")
    args.add_argument('--path', type=str, required=True)
    args.add_argument('--label_path', type=str, required=True)
    args.add_argument('--image_size', type=int, default=128)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--workers', type=int, default=0)
    args.add_argument('--shuffle', action='store_true')
    # args.add_argument()

    args.add_argument('--latent_size', type=int, default=120)
    opt = args.parse_args()

    base_tfs, additional_tfs = get_transforms(image_size=128)
    ds = MotorbikeWithLabelsDataset(opt.path, opt.label_path, base_tfs, additional_tfs, in_memory=True)
    dl = MotorbikeDataloader(opt, ds)

    a, b = dl.next_batch()
    print(a.shape)
    print(b.shape)

    assert a.shape[0]  == opt.batch_size
    assert a.shape[2] == a.shape[3]
    assert a.shape[3] == opt.image_size
    assert b.shape[0] == opt.batch_size

    print(dl.dataset.class_dist)
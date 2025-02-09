import copy

import cv2
import numpy as np
import torch
from RandAugment import RandAugment
from torch.utils.data import Dataset, SubsetRandomSampler
import os

from config.const import data_classes
from utils import Domain, read_image_by_path


class ImageSet(object):

    def __init__(self, data, labels):
        """
        :param data: path list
        :param labels: labels list
        """
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return cv2.imread(self.data[index]), self.labels[index]

    def __len__(self):
        return len(self.data)


class AugImageSet(Dataset):

    def __init__(self, dataset: ImageSet, transforms, target_transforms):
        self.data = dataset.data
        self.labels = dataset.labels
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.rand_aug_transforms = copy.deepcopy(self.target_transforms)
        self.committee_size = 1
        self.ra_obj = RandAugment(1, 2.0)
        self.rand_aug_transforms.transforms.insert(0, self.ra_obj)

    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]
        data = read_image_by_path(self.data[index])
        rand_aug_lst = [self.rand_aug_transforms(data) for _ in range(self.committee_size)]
        return (self.transforms(data), self.target_transforms(data), rand_aug_lst), int(label), int(index)

    def __len__(self):
        return len(self.data)


class Office31(object):
    def __init__(self, classes, path, train_transforms, test_transforms, domain: Domain, enable=False, batch_size=128):
        self.classes = classes
        self.train = None
        self.test = None
        self.val = None
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.path = path
        self.domain = domain
        self.enable = enable
        self.batch_size = batch_size
        self.data = []
        self.labels = []
        self.load_data()

    def load(self):
        train_dataset = ImageSet(data=self.data, labels=np.array(self.labels))
        test_dataset = ImageSet(data=self.data, labels=np.array(self.labels))
        val_dataset = ImageSet(data=self.data, labels=np.array(self.labels))
        train_dataset.labels, test_dataset.labels, val_dataset.labels = torch.from_numpy(
            train_dataset.labels), torch.from_numpy(test_dataset.labels), torch.from_numpy(val_dataset.labels)
        self.train = AugImageSet(train_dataset, self.train_transforms, self.test_transforms)
        self.test = AugImageSet(test_dataset, self.train_transforms, self.test_transforms)
        self.val = AugImageSet(val_dataset, self.train_transforms, self.test_transforms)
        return self.train, self.test, self.val

    def loader(self, shuffle=True, num_workers=4, class_balance_train=False):
        if not self.train:
            self.load()
        num_train = len(self.train)
        train_size = num_train
        train_idx = np.arange(len(self.train))
        if class_balance_train:
            self.train.data = [self.train.data[idx] for idx in train_idx]
            self.train.labels = self.train.labels[train_idx]
            train_sampler = None
        else:
            train_sampler = SubsetRandomSampler(train_idx)
        train_loader = torch.utils.data.DataLoader(self.train, sampler=train_sampler,
                                                   batch_size=self.batch_size, num_workers=num_workers, drop_last=True, shuffle=False)
        val_loader = torch.utils.data.DataLoader(self.val, shuffle=False,
                                                 batch_size=self.batch_size)
        test_loader = torch.utils.data.DataLoader(self.test, batch_size=self.batch_size)
        return train_loader, val_loader, test_loader, train_idx

    def load_data(self):
        path = f'{self.path}/{self.domain.name}'
        for cls in self.domain.classes:
            data = os.listdir(f'{path}/{cls}')
            self.data.extend([f'{path}/{cls}/{d}' for d in data])
            self.labels.extend([data_classes[cls]] * len(data))
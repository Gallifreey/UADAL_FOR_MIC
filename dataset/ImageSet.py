from torch.utils.data import Dataset
import os

from utils import Domain


class Office31(Dataset):

    def __init__(self, path, root_cls, domain: Domain):
        self.root = path
        self.classes = root_cls
        self.domain = domain
        self.data = []
        self.__preprocess()

    def __preprocess(self):
        for c in self.domain.cls:
            datas = os.listdir(f'{self.root}/{self.domain.name}/{c}/')
            for data in datas:
                self.data.append((
                    f'{self.root}/{self.domain.name}/{c}/{data}',
                    self.classes[c]
                ))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

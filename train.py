import torch
from torch.utils.data import DataLoader

from config.const import root, cls
from dataset.ImageSet import Office31
from utils import Domain, list2dict


if __name__ == '__main__':
    source = Domain('amazon', cls=list2dict([0, 1, 2, 3, 4, 5], cls))
    target = Domain('dslr', cls=list2dict([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cls))
    office31_source = Office31(path=root,
                               root_cls=cls,
                               domain=source)
    office31_target = Office31(path=root,
                               root_cls=cls,
                               domain=target)
    office31_source_loader = DataLoader(office31_source, shuffle=True)
    office31_target_loader = DataLoader(office31_target, shuffle=True)


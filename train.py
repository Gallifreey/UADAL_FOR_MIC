import random
import warnings

import numpy as np
import torch

from config import args
from config.const import root, data_classes
from dataset.ImageSet import Office31
from models.UADAL_MIC import UADAL_MIC
from utils import Domain, list2dict, get_test_train_transform
warnings.filterwarnings("ignore")

def start(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:%s" % 0)

    source = Domain('amazon', list2dict([0, 1, 2, 3, 4, 5], data_classes))
    target = Domain('dslr', list2dict([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], data_classes))
    train_transform, test_transform = get_test_train_transform()
    office31_source = Office31(classes=data_classes,
                               path=root,
                               domain=source,
                               train_transforms=train_transform,
                               test_transforms=test_transform)
    office31_source.load()
    office31_target = Office31(classes=data_classes,
                               path=root,
                               domain=target,
                               train_transforms=train_transform,
                               test_transforms=test_transform)
    office31_target.load()
    model = UADAL_MIC(args, len(data_classes), office31_source, office31_target)
    model.init()
    model.warmup()
    model.test(0)
    model.build()
    model.train()


if __name__ == '__main__':
    start(args.args)

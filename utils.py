import math


def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.weight.data.fill_(1.)


def list2dict(arr: list, dic: dict):
    res = []
    for a in arr:
        for key, val in dic.items():
            if val == a:
                res.append(key)
    return res


class Domain(object):

    def __init__(self, name, cls):
        self.name = name
        self.cls = cls

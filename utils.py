import math
import sys

import cv2
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import scipy.stats as stats
from PIL import Image
from matplotlib import pyplot as plt
from scipy.special import beta as beta_f
from torchvision.transforms import transforms

from models.VGGish.vggish import VGGish
from models.backbone import ResNetFc, EfficientNetB0, DenseNet
from models.base import Recognizer, Classifier, Discriminator


def list2dict(arr: list, dic: dict):
    res = []
    for a in arr:
        for key, val in dic.items():
            if val == a:
                res.append(key)
    return res


def InverseDecayScheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))


def ConstantScheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    return initial_lr


def CosineScheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    cos = (1 + np.cos((step / max_iter) * np.pi)) / 2
    lr = initial_lr * cos
    return lr


def StepScheduler(step, initial_lr, gamma=500, power=0.2, max_iter=1000):
    divide = step // 500
    lr = initial_lr * (0.2 ** divide)
    return lr


def init_base_model(args, known_num_class, all_num_class):
    backbone = args.backbone
    bottle_neck_dim = args.bottle_neck_dim
    if "vggish" in backbone:
        model_G = VGGish(pretrained=False)
    elif 'resnet' in backbone:
        model_G = ResNetFc(model_name=args.backbone)
    elif 'effi' in backbone:
        model_G = EfficientNetB0()
    elif 'densenet' in backbone:
        model_G = DenseNet()
    else:
        print('Please specify the backbone network')
        sys.exit()
    model_R = Recognizer(in_dim=model_G.output_num(), out_dim=known_num_class, bias=False)
    model_C = Classifier(in_dim=model_G.output_num(), out_dim=all_num_class, bottle_neck_dim=bottle_neck_dim)
    return model_G, model_R, model_C


def init_deep_model(args, known_num_class, all_num_class, domain_dim=3, dc_out_dim=None):
    backbone = args.backbone
    bottle_neck_dim = args.bottle_neck_dim
    bottle_neck_dim2 = args.bottle_neck_dim2
    if "vggish" in backbone:
        model_G = VGGish(pretrained=False)
    elif 'resnet' in backbone:
        model_G = ResNetFc(model_name=args.backbone)
    elif 'effi' in backbone:
        model_G = EfficientNetB0()
    elif 'densenet' in backbone:
        model_G = DenseNet()
    else:
        print('Please specify the backbone network')
        sys.exit()
    if dc_out_dim is None:
        dc_dim = model_G.output_num()
    else:
        dc_dim = dc_out_dim

    model_R = Recognizer(in_dim=model_G.output_num(), out_dim=known_num_class, bias=False)
    model_C = Classifier(in_dim=model_G.output_num(), out_dim=all_num_class, bottle_neck_dim=bottle_neck_dim)
    model_D = Discriminator(dc_dim, out_dim=domain_dim, bottle_neck_dim=bottle_neck_dim2)

    return model_G, model_R, model_C, model_D


class Domain(object):

    def __init__(self, name, classes):
        self.name = name
        self.classes = classes  # [0, 1, 2, 3, ..]


class OptimWithScheduler:
    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.global_step = 0.0
        for g in self.optimizer.param_groups:
            g['initial_lr'] = g['lr']

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler_func(step=self.global_step, initial_lr=g['initial_lr'])
        self.optimizer.step()
        self.global_step += 1


def CrossEntropyLoss(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()

    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
        instance_normalize = N
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        instance_normalize = torch.sum(instance_level_weight) + epsilon
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    ce = -label * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * ce * class_level_weight) / float(instance_normalize)


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1)
        return b


def extended_confusion_matrix(y_true, y_pred, true_labels=None, pred_labels=None):
    if not true_labels:
        true_labels = sorted(list(set(list(y_true))))
    true_label_to_id = {x: i for (i, x) in enumerate(true_labels)}
    if not pred_labels:
        pred_labels = true_labels
    pred_label_to_id = {x: i for (i, x) in enumerate(pred_labels)}
    confusion_matrix = np.zeros([len(true_labels), len(pred_labels)])
    for (true, pred) in zip(y_true, y_pred):
        confusion_matrix[true_label_to_id[true]][pred_label_to_id[pred]] += 1.0
    return confusion_matrix


def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)


def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar) ** 2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) / x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=None,
                 betas_init=None,
                 weights_init=None):
        self.criteria = None
        self.K = None
        self.score_history = None
        self.weight_0 = None
        self.weight_1 = None
        if weights_init is None:
            weights_init = [0.5, 0.5]
        if betas_init is None:
            betas_init = [2, 1]
        if alphas_init is None:
            alphas_init = [1, 2]
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):  # p(k)*p(l|k) == p(y)*p(x|y)
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r = np.array([self.weighted_likelihood(x, i) for i in range(2)])
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps
        self.score_history = []
        self.weight_0 = []
        self.weight_1 = []
        for i in range(self.max_iters):
            # E-step
            r = self.responsibilities(x)
            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

            neg_log_likelihood = np.sum([self.score_samples(i) for i in x])
            self.score_history.append(neg_log_likelihood)
            self.weight_0.append(self.weight[0])
            self.weight_1.append(self.weight[1])
        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0 + self.eps_nan, 1 - self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l

    def look_lookup(self, x, loss_max, loss_min, testing=False):
        if testing:
            x_i = x
        else:
            x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self, title, save_dir, save_signal=False):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label='known')
        plt.plot(x, self.weighted_likelihood(x, 1), label='unknown')
        plt.plot(x, self.probability(x), lw=2, label='mixture')
        plt.legend()
        if save_signal:
            plt.title(title)
            plt.savefig(save_dir, dpi=300)
        plt.close()

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)

    def calculate_criteria(self):
        self.K = (self.weight[0] * beta_f(self.alphas[1], self.betas[1])) / (
                self.weight[1] * beta_f(self.alphas[0], self.betas[0]))
        self.criteria = ((np.log(self.K)) - (self.betas[1] - self.betas[0])) / (
                (self.alphas[1] - self.alphas[0]) - (self.betas[1] - self.betas[0]))
        print(self.K, self.alphas[1] - self.alphas[0], beta_f(2, 3))
        return self.criteria


def get_test_train_transform():
    normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_transform
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize_transform
    ])

    return train_transforms, test_transforms


def read_image_by_path(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

import copy

import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd.variable import Variable

from utils import init_base_model, InverseDecayScheduler, ConstantScheduler, CosineScheduler, StepScheduler, \
    OptimWithScheduler, init_deep_model, CrossEntropyLoss, HLoss, extended_confusion_matrix, BetaMixture1D


class UADAL_MIC(object):

    def __init__(self, args, num_class, src_dset, target_dset):
        super().__init__()
        self.args = args
        self.all_num_class = num_class
        self.known_num_class = num_class - 1
        self.src_dset = src_dset
        self.target_dset = target_dset
        self.device = self.args.device
        self.init()
        self.ent_criterion = HLoss()
        self.bmm_model = self.cont = self.k = 0
        self.bmm_model_maxLoss = torch.log(torch.FloatTensor([self.known_num_class])).to(self.device)
        self.bmm_model_minLoss = torch.FloatTensor([0.0]).to(self.device)
        self.bmm_update_cnt = 0
        self.G = None
        self.R = None
        self.C = None
        self.D = None
        self.scheduler_g = None
        self.scheduler_r = None
        self.scheduler_c = None
        self.scheduler_d = None
        self.G_frozen = None
        self.R_frozen = None
        self.src_train_loader, self.src_val_loader, self.src_test_loader, self.src_train_idx = src_dset.loader(class_balance_train=False)
        self.target_train_loader, self.target_val_loader, self.target_test_loader, self.tgt_train_idx = target_dset.loader()
        self.num_batches = min(len(self.src_train_loader), len(self.target_train_loader))

    def init(self):
        self.G, self.R, self.C = init_base_model(self.args, self.known_num_class, self.all_num_class)
        if self.args.cuda:
            self.G.to(self.device)
            self.R.to(self.device)
            self.C.to(self.device)
        scheduler = lambda step, initial_lr: InverseDecayScheduler(step, initial_lr, gamma=0, power=0.75,
                                                                   max_iter=self.args.warmup_iter)
        params = list(self.G.parameters())
        self.scheduler_g = OptimWithScheduler(
            optim.SGD(params, lr=self.args.g_lr * self.args.e_lr, weight_decay=5e-4, momentum=0.9,
                      nesterov=True), scheduler)
        self.scheduler_r = OptimWithScheduler(
            optim.SGD(self.R.parameters(), lr=self.args.e_lr, weight_decay=5e-4, momentum=0.9, nesterov=True),
            scheduler)
        self.scheduler_c = OptimWithScheduler(
            optim.SGD(self.C.parameters(), lr=self.args.e_lr, weight_decay=5e-4, momentum=0.9, nesterov=True),
            scheduler)

    @staticmethod
    def weights_init_bias_zero(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)

    def frozen_GR(self):
        self.G_frozen = copy.deepcopy(self.G)
        self.R_frozen = copy.deepcopy(self.R)

    def compute_probabilities_batch(self, out_t, unk=1):
        ent_t = self.ent_criterion(out_t)
        batch_ent_t = (ent_t - self.bmm_model_minLoss) / (self.bmm_model_maxLoss - self.bmm_model_minLoss + 1e-6)
        batch_ent_t[batch_ent_t >= 1 - 1e-4] = 1 - 1e-4
        batch_ent_t[batch_ent_t <= 1e-4] = 1e-4
        B = self.bmm_model.posterior(batch_ent_t.clone().cpu().numpy(), unk)
        B = torch.FloatTensor(B)
        return B

    def reset_network(self):
        if 'resnet' in self.args.net:
            try:
                self.R.fc.reset_parameters()
                self.R.bottleneck.reset_parameters()
            except:
                self.R.fc.reset_parameters()
        elif 'vgg' in self.args.net:
            try:
                self.R.fc.reset_parameters()
                self.R.bottleneck.reset_parameters()
            except:
                self.R.fc.reset_parameters()

    def build(self):
        _, self.R, _, self.D = init_deep_model(self.args, known_num_class=self.known_num_class,
                                               all_num_class=self.all_num_class, domain_dim=3)
        self.D.apply(self.weights_init_bias_zero)
        if self.args.cuda:
            self.R.to(self.device)
            self.D.to(self.device)
        SCHEDULER = {
            'cos': CosineScheduler,
            'step': StepScheduler,
            'id': InverseDecayScheduler,
            'constant': ConstantScheduler
        }
        scheduler = lambda step, initial_lr: SCHEDULER[self.args.scheduler](step, initial_lr, gamma=10, power=0.75,
                                                                            max_iter=self.num_batches * self.args.training_iter)
        scheduler_d = lambda step, initial_lr: SCHEDULER[self.args.scheduler](step, initial_lr, gamma=10, power=0.75,
                                                                              max_iter=self.num_batches * self.args.training_iter * self.args.update_freq_D)
        scheduler_r = lambda step, initial_lr: InverseDecayScheduler(step, initial_lr, gamma=0, power=0.75,
                                                                     max_iter=self.num_batches * self.args.training_iter)
        params = list(self.G.parameters())

        self.scheduler_g = OptimWithScheduler(
            optim.SGD(params, lr=self.args.g_lr * self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True),
            scheduler)
        self.scheduler_c = OptimWithScheduler(
            optim.SGD(self.C.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)
        self.scheduler_d = OptimWithScheduler(
            optim.SGD(self.D.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True),
            scheduler_d)
        self.scheduler_r = OptimWithScheduler(
            optim.SGD(self.R.parameters(), lr=self.args.e_lr, weight_decay=5e-4, momentum=0.9, nesterov=True),
            scheduler_r)

    def warmup(self):
        print("==========START WARMUP PROCESS==========")
        epoch_cnt = 0
        step = 0
        while step < self.args.warmup_iter + 1:
            self.G.train()
            self.R.train()
            self.C.train()
            epoch_cnt += 1
            for batch_idx, ((_, _, aud_s_aug), label_s, _) in enumerate(self.src_train_loader):
                aud_s = aud_s_aug[0]
                if self.args.cuda:
                    aud_s = Variable(aud_s.to(self.device))
                    label_s = Variable(label_s.to(self.device))
                step += 1
                if step >= self.args.warmup_iter + 1:
                    break
                self.scheduler_g.zero_grad()
                self.scheduler_r.zero_grad()
                self.scheduler_c.zero_grad()

                feat_s = self.G(aud_s)
                out_s = self.R(feat_s)

                label_s_onehot = nn.functional.one_hot(label_s, num_classes=self.known_num_class)
                label_s_onehot = label_s_onehot * (1 - self.args.ls_eps)
                label_s_onehot = label_s_onehot + self.args.ls_eps / self.known_num_class
                loss_s = CrossEntropyLoss(label=label_s_onehot, predict_prob=F.softmax(out_s, dim=1))

                out_Cs = self.C(feat_s)
                label_s_onehot = nn.functional.one_hot(label_s, num_classes=self.all_num_class)
                label_s_onehot = label_s_onehot * (1 - self.args.ls_eps)
                label_s_onehot = label_s_onehot + self.args.ls_eps / self.all_num_class
                loss_Cs = CrossEntropyLoss(label=label_s_onehot, predict_prob=F.softmax(out_Cs, dim=1))

                loss = loss_s + loss_Cs

                loss.backward()
                self.scheduler_g.step()
                self.scheduler_r.step()
                self.scheduler_c.step()
                self.scheduler_g.zero_grad()
                self.scheduler_r.zero_grad()
                self.scheduler_c.zero_grad()
                if step % self.args.show_step == 0:
                    print(f"Epoch {epoch_cnt} Step {step} Loss {loss.item()}")
        print("==========WARMUP PROCESS FINISHED==========")

    def train(self):
        print("==========START TRAINING PROCESS==========")
        for epoch in range(1, self.args.training_iter):
            joint_loader = zip(self.src_train_loader, self.target_train_loader)
            alpha = float((float(2) / (1 + np.exp(-10 * float((float(epoch) / float(self.args.training_iter)))))) - 1)
            for batch_idx, (((aud_s, _, _), label_s, _), ((aud_t, aud_t_og, aud_t_aug), label_t, index_t)) in enumerate(
                    joint_loader):
                self.G.train()
                self.C.train()
                self.D.train()
                self.R.train()
                if self.args.cuda:
                    aud_s = Variable(aud_s.to(self.device))
                    label_s = Variable(label_s.to(self.device))
                    aud_t = Variable(aud_t.to(self.device))
                    aud_t_og = Variable(aud_t_og.to(self.device))
                    aud_t_aug = Variable(aud_t_aug[0].to(self.device))

                out_t_free = self.R_frozen(self.G_frozen(aud_t_og)).detach()
                w_unk_posterior = self.compute_probabilities_batch(out_t_free, 1)
                w_k_posterior = 1 - w_unk_posterior
                w_k_posterior = w_k_posterior.to(self.device)
                w_unk_posterior = w_unk_posterior.to(self.device)

                for d_step in range(self.args.update_freq_D):
                    self.scheduler_d.zero_grad()
                    feat_s = self.G(aud_s).detach()
                    out_ds = self.D(feat_s)
                    label_ds = Variable(torch.zeros(aud_s.size()[0], dtype=torch.long).to(self.device))
                    label_ds = nn.functional.one_hot(label_ds, num_classes=3)
                    loss_ds = CrossEntropyLoss(label=label_ds, predict_prob=F.softmax(out_ds, dim=1))

                    label_dt_known = Variable(torch.ones(aud_t.size()[0], dtype=torch.long).to(self.device))
                    label_dt_known = nn.functional.one_hot(label_dt_known, num_classes=3)
                    label_dt_unknown = 2 * Variable(torch.ones(aud_t.size()[0], dtype=torch.long).to(self.device))
                    label_dt_unknown = nn.functional.one_hot(label_dt_unknown, num_classes=3)

                    feat_t = self.G(aud_t).detach()
                    out_dt = self.D(feat_t)

                    label_dt = w_k_posterior[:, None] * label_dt_known + w_unk_posterior[:, None] * label_dt_unknown
                    loss_dt = CrossEntropyLoss(label=label_dt, predict_prob=F.softmax(out_dt, dim=1))
                    loss_D = 0.5 * (loss_ds + loss_dt)
                    loss_D.backward()

                    if self.args.opt_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.args.opt_clip)
                    self.scheduler_d.step()
                    self.scheduler_d.zero_grad()

                for _ in range(self.args.update_freq_G):
                    self.scheduler_g.zero_grad()
                    self.scheduler_c.zero_grad()
                    self.scheduler_r.zero_grad()
                    feat_s = self.G(aud_s)
                    out_ds = self.D(feat_s)
                    loss_ds = CrossEntropyLoss(label=label_ds, predict_prob=F.softmax(out_ds, dim=1))
                    feat_t = self.G(aud_t)
                    out_dt = self.D(feat_t)
                    label_dt = w_k_posterior[:, None] * label_dt_known - w_unk_posterior[:, None] * label_dt_unknown
                    loss_dt = CrossEntropyLoss(label=label_dt, predict_prob=F.softmax(out_dt, dim=1))
                    loss_G = alpha * (- loss_ds - loss_dt)

                    out_Rs = self.R(feat_s)
                    label_s_onehot = nn.functional.one_hot(label_s, num_classes=self.known_num_class)
                    label_s_onehot = label_s_onehot * (1 - self.args.ls_eps)
                    label_s_onehot = label_s_onehot + self.args.ls_eps / self.known_num_class
                    loss_cls_Rs = CrossEntropyLoss(label=label_s_onehot,
                                                   predict_prob=F.softmax(out_Rs, dim=1))
                    out_Cs = self.C(feat_s)
                    label_Cs_onehot = nn.functional.one_hot(label_s, num_classes=self.all_num_class)
                    label_Cs_onehot = label_Cs_onehot * (1 - self.args.ls_eps)
                    label_Cs_onehot = label_Cs_onehot + self.args.ls_eps / self.all_num_class
                    loss_cls_Cs = CrossEntropyLoss(label=label_Cs_onehot, predict_prob=F.softmax(out_Cs, dim=1))

                    label_unknown = self.known_num_class * Variable(
                        torch.ones(aud_t.size()[0], dtype=torch.long).to(self.device))
                    label_unknown = nn.functional.one_hot(label_unknown, num_classes=self.all_num_class)
                    label_unknown_lsr = label_unknown * (1 - self.args.ls_eps)
                    label_unknown_lsr = label_unknown_lsr + self.args.ls_eps / self.all_num_class

                    feat_t_aug = self.G(aud_t_aug)
                    out_Ct = self.C(feat_t)
                    out_Ct_aug = self.C(feat_t_aug)

                    loss_cls_Ctu = alpha * CrossEntropyLoss(label=label_unknown_lsr,
                                                            predict_prob=F.softmax(out_Ct_aug, dim=1),
                                                            instance_level_weight=w_unk_posterior)
                    pseudo_label = torch.softmax(out_Ct.detach(), dim=-1)
                    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                    targets_u_onehot = nn.functional.one_hot(targets_u, num_classes=self.all_num_class)
                    mask = max_probs.ge(self.args.threshold).float()
                    loss_ent_Ctk = CrossEntropyLoss(label=targets_u_onehot,
                                                    predict_prob=F.softmax(out_Ct_aug, dim=1),
                                                    instance_level_weight=mask)
                    loss = loss_cls_Rs + loss_cls_Cs + 0.5 * loss_G + 0.5 * loss_ent_Ctk + 0.2 * loss_cls_Ctu
                    loss.backward()
                    self.scheduler_g.step()
                    self.scheduler_c.step()
                    self.scheduler_r.step()
                    self.scheduler_g.zero_grad()
                    self.scheduler_c.zero_grad()
                    self.scheduler_r.zero_grad()

            if epoch % self.args.update_term == 0:
                C_acc_os, C_acc_os_star, C_acc_unknown, C_acc_hos = self.test(epoch)
                print('Epoch_{:>3}/{:>3}_OS_{:.3f}_OS*_{:.3f}_UNK_{:.3f}_HOS_{:.3f}_BMM{}'.format(
                        epoch, self.args.training_iter, C_acc_os, C_acc_os_star, C_acc_unknown, C_acc_hos, self.bmm_update_cnt))
        C_acc_os, C_acc_os_star, C_acc_unknown, C_acc_hos = self.test(self.args.training_iter)
        print('Epoch_{:>3}/{:>3}_OS_{:.3f}_OS*_{:.3f}_UNK_{:.3f}_HOS_{:.3f}_BMM{}'.format(
                self.args.training_iter, self.args.training_iter, C_acc_os, C_acc_os_star, C_acc_unknown, C_acc_hos, self.bmm_update_cnt))
        print("==========TRAINING PROCESS FINISHED==========")

    def test(self, epoch):
        self.G.eval()
        self.C.eval()
        self.R.eval()
        total_pred_t = np.array([])
        total_label_t = np.array([])
        all_ent_t = torch.Tensor([])
        with torch.no_grad():
            for batch_idx, ((aud_t, _, _), label_t, index_t) in enumerate(self.target_test_loader):
                if self.args.cuda:
                    aud_t, label_t = Variable(aud_t.to(self.device)), Variable(label_t.to(self.device))
                feat_t = self.G(aud_t)
                out_t = F.softmax(self.C(feat_t), dim=1)

                pred = out_t.data.max(1)[1]
                pred_numpy = pred.cpu().numpy()
                total_pred_t = np.append(total_pred_t, pred_numpy)
                total_label_t = np.append(total_label_t, label_t.cpu().numpy())

                out_Et = self.R(feat_t)
                ent_Et = self.ent_criterion(out_Et)
                all_ent_t = torch.cat((all_ent_t, ent_Et.cpu()))

        max_target_label = int(np.max(total_label_t) + 1)
        m = extended_confusion_matrix(total_label_t, total_pred_t, true_labels=list(range(max_target_label)),
                                      pred_labels=list(range(self.all_num_class)))
        cm = m
        cm = cm.astype(np.float) / np.sum(cm, axis=1, keepdims=True)
        acc_os_star = sum([cm[i][i] for i in range(self.known_num_class)]) / self.known_num_class
        acc_unknown = sum(
            [cm[i][self.known_num_class] for i in range(self.known_num_class, int(np.max(total_label_t) + 1))]) / (
                              max_target_label - self.known_num_class)
        acc_os = (acc_os_star * self.known_num_class + acc_unknown) / (self.known_num_class + 1)
        acc_hos = (2 * acc_os_star * acc_unknown) / (acc_os_star + acc_unknown)

        self.G.train()
        self.C.train()
        self.R.train()

        if epoch % self.args.update_term == 0:
            entropy_list = all_ent_t.data.numpy()
            loss_tr_t = (entropy_list - self.bmm_model_minLoss.data.cpu().numpy()) / (
                    self.bmm_model_maxLoss.data.cpu().numpy() - self.bmm_model_minLoss.data.cpu().numpy() + 1e-6)
            loss_tr_t[loss_tr_t >= 1] = 1 - 10e-4
            loss_tr_t[loss_tr_t <= 0] = 10e-4
            self.bmm_model = BetaMixture1D()
            self.bmm_model.fit(loss_tr_t)
            self.bmm_model.create_lookup(1)
            self.bmm_update_cnt += 1
            self.frozen_GR()
            self.reset_network()
            return acc_os, acc_os_star, acc_unknown, acc_hos

    def validate(self):
        pass

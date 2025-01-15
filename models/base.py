import torch.nn as nn
import torch.nn.functional as F
import torch

from utils import init_layer, init_bn


class Attention(nn.Module):
    def __init__(self, n_in, n_out):
        super(Attention, self).__init__()

        self.att = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att, )
        init_layer(self.cla)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """
        att = self.att(x)
        att = torch.sigmoid(att)

        cla = self.cla(x)
        cla = torch.sigmoid(cla)

        att = att[:, :, :, 0]  # (samples_num, classes_num, time_steps)
        cla = cla[:, :, :, 0]  # (samples_num, classes_num, time_steps)

        epsilon = 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)

        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        x = torch.sum(norm_att * cla, dim=2)

        x = F.hardtanh(x, 0., 1.)
        return x


class EmbeddingLayers(nn.Module):

    def __init__(self, freq_bins, emb_layers, hidden_units, drop_rate):
        super(EmbeddingLayers, self).__init__()
        self.freq_bins = freq_bins
        self.hidden_units = hidden_units
        self.drop_rate = drop_rate

        self.conv1x1 = nn.ModuleList()
        self.batchnorm = nn.ModuleList()

        for i in range(emb_layers):
            in_channels = freq_bins if i == 0 else hidden_units
            conv = nn.Conv2d(
                in_channels=in_channels, out_channels=hidden_units,
                kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
            self.conv1x1.append(conv)
            self.batchnorm.append(nn.BatchNorm2d(in_channels))

        # Append last batch-norm layer
        self.batchnorm.append(nn.BatchNorm2d(hidden_units))

        self.init_weights()

    def init_weights(self):

        for conv in self.conv1x1:
            init_layer(conv)

        for bn in self.batchnorm:
            init_bn(bn)

    def forward(self, input, return_layers=False):
        """input: (samples_num, time_steps, freq_bins)
        """

        drop_rate = self.drop_rate

        # (samples_num, freq_bins, time_steps)
        x = input.transpose(1, 2)

        # Add an extra dimension for using Conv2d
        # (samples_num, freq_bins, time_steps, 1)

        x = x[:, :, :, None].contiguous()

        out = self.batchnorm[0](x)
        residual = x
        all_outs = [out]

        for i in range(len(self.conv1x1)):
            out = F.dropout(F.relu(self.batchnorm[i + 1](self.conv1x1[i](out))),
                            p=drop_rate,
                            training=self.training)
            all_outs.append(out)

        if return_layers is False:
            # (samples_num, hidden_units, time_steps, 1)
            return out

        else:
            return all_outs


class DecisionLevelSingleAttention(nn.Module):

    def __init__(self, freq_bins, classes_num, emb_layers, hidden_units, drop_rate):
        super(DecisionLevelSingleAttention, self).__init__()

        self.emb = EmbeddingLayers(
            freq_bins=freq_bins,
            emb_layers=emb_layers,
            hidden_units=hidden_units,
            drop_rate=drop_rate)

        self.attention = Attention(
            n_in=hidden_units,
            n_out=classes_num)

    def init_weights(self):
        pass

    def forward(self, input):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        # (samples_num, hidden_units, time_steps, 1)
        b1 = self.emb(input)

        # (samples_num, classes_num, time_steps, 1)
        output = self.attention(b1)

        return output


class Recognizer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(Recognizer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        x = self.fc(x)
        return x


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim, bias=True):
        super(Classifier, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim, bias=bias)
        self.main = nn.Sequential(self.bottleneck,
                                  nn.Sequential(nn.BatchNorm1d(bottle_neck_dim), nn.LeakyReLU(0.2, inplace=True),
                                                self.fc))

    def forward(self, x):
        for module in self.main.children():
            x = module(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim=None):
        super(Discriminator, self).__init__()
        if bottle_neck_dim is None:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(
                self.fc,
            )
        else:
            self.main = nn.Sequential(nn.Linear(in_dim, bottle_neck_dim), nn.LeakyReLU(0.2, inplace=True),
                                      nn.Linear(bottle_neck_dim, bottle_neck_dim), nn.LeakyReLU(0.2, inplace=True),
                                      nn.Linear(bottle_neck_dim, out_dim))

    def forward(self, x):
        for module in self.main.children():
            x = module(x)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

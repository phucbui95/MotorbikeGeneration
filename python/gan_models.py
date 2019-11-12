import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm_notebook
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import math


from dataset import MotorbikeDataset, MotorbikeWithLabelsDataset, get_transforms
from layers import init_weight, conv1x1, conv3x3, Attention, ConditionalNorm


# BigGAN + leaky_relu
class ResBlock_G(nn.Module):
    def __init__(self, in_channel, out_channel, condition_dim, upsample=True):
        super().__init__()
        self.cbn1 = ConditionalNorm(in_channel, condition_dim)
        self.upsample = nn.Sequential()
        if upsample:
            self.upsample.add_module('upsample', nn.Upsample(scale_factor=2,
                                                             mode='nearest'))
        self.conv3x3_1 = nn.utils.spectral_norm(
            conv3x3(in_channel, out_channel)).apply(init_weight)
        self.cbn2 = ConditionalNorm(out_channel, condition_dim)
        self.conv3x3_2 = nn.utils.spectral_norm(
            conv3x3(out_channel, out_channel)).apply(init_weight)
        self.conv1x1 = nn.utils.spectral_norm(
            conv1x1(in_channel, out_channel)).apply(init_weight)

    def forward(self, inputs, condition):
        x = F.leaky_relu(self.cbn1(inputs, condition))
        x = self.upsample(x)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(F.leaky_relu(self.cbn2(x, condition)))
        x += self.conv1x1(self.upsample(inputs))  # shortcut
        return x


class Generator(nn.Module):
    def __init__(self, n_feat, codes_dim, n_classes=0):
        super().__init__()
        self.codes_dim = codes_dim
        self.fc = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Linear(codes_dim, 16 * n_feat * 4 * 4)).apply(init_weight)
        )
        self.res1 = ResBlock_G(16 * n_feat, 16 * n_feat, codes_dim + n_classes,
                               upsample=True)
        self.res2 = ResBlock_G(16 * n_feat, 8 * n_feat, codes_dim + n_classes,
                               upsample=True)
        self.res3 = ResBlock_G(8 * n_feat, 4 * n_feat, codes_dim + n_classes,
                               upsample=True)
        self.attn = Attention(2 * n_feat)
        self.res4 = ResBlock_G(4 * n_feat, 2 * n_feat, codes_dim + n_classes,
                               upsample=True)
        self.res5 = ResBlock_G(2 * n_feat, n_feat, codes_dim + n_classes,
                               upsample=True)
        self.conv = nn.Sequential(
            # nn.BatchNorm2d(2*n_feat).apply(init_weight),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(conv3x3(n_feat, 3)).apply(init_weight),
        )

    def forward(self, z, label_ohe):
        '''
        z.shape = (*,Z_DIM)
        cd.shape = (*,n_classes)
        '''
        batch = z.size(0)
        z = z.squeeze()
        label_ohe = label_ohe.squeeze()
        codes = torch.split(z, self.codes_dim, dim=1)
        print(codes[0].shape)

        x = self.fc(codes[0])  # ->(*,16ch*4*4)
        x = x.view(batch, -1, 4, 4)  # ->(*,16ch,4,4)

        condition = torch.cat([codes[1], label_ohe],
                              dim=1)  # (*,codes_dim+n_classes)
        x = self.res1(x, condition)  # ->(*,16ch,8,8)

        condition = torch.cat([codes[2], label_ohe], dim=1)
        x = self.res2(x, condition)  # ->(*,8ch,16,16)

        condition = torch.cat([codes[3], label_ohe], dim=1)
        x = self.res3(x, condition)  # ->(*,4ch,32,32)

        condition = torch.cat([codes[4], label_ohe], dim=1)
        x = self.res4(x, condition)  # ->(*,2ch,64,64)
        #         x = self.attn(x) #not change shape

        condition = torch.cat([codes[5], label_ohe], dim=1)
        x = self.res5(x, condition)  # ->(*,2ch,128,128)

        x = self.conv(x)  # ->(*,3,128,128)
        x = torch.tanh(x)
        return x


class ResBlock_D(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(conv3x3(in_channel, out_channel)).apply(
                init_weight),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(conv3x3(out_channel, out_channel)).apply(
                init_weight),
        )
        self.shortcut = nn.Sequential(
            nn.utils.spectral_norm(conv1x1(in_channel, out_channel)).apply(
                init_weight),
        )
        if downsample:
            self.layer.add_module('avgpool',
                                  nn.AvgPool2d(kernel_size=2, stride=2))
            self.shortcut.add_module('avgpool',
                                     nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, inputs):
        x = self.layer(inputs)
        x += self.shortcut(inputs)
        return x


class Discriminator(nn.Module):
    def __init__(self, n_feat, n_classes=0):
        super().__init__()
        self.res1 = ResBlock_D(3, n_feat, downsample=True)
        self.attn = Attention(n_feat)
        self.res2 = ResBlock_D(n_feat, 2 * n_feat, downsample=True)
        self.res3 = ResBlock_D(2 * n_feat, 4 * n_feat, downsample=True)
        self.res4 = ResBlock_D(4 * n_feat, 8 * n_feat, downsample=True)
        self.res5 = ResBlock_D(8 * n_feat, 16 * n_feat, downsample=True)
        self.res6 = ResBlock_D(16 * n_feat, 16 * n_feat, downsample=False)

        self.fc = nn.utils.spectral_norm(nn.Linear(16 * n_feat, 1)).apply(
            init_weight)
        self.embedding = nn.Embedding(num_embeddings=n_classes,
                                      embedding_dim=16 * n_feat).apply(
            init_weight)

    def forward(self, inputs, label):
        batch = inputs.size(0)  # (*,3,128,128)
        h = self.res1(inputs)  # ->(*,ch,64,64)
        #         h = self.attn(h) #not change shape
        h = self.res2(h)  # ->(*,2ch,32,32)
        h = self.res3(h)  # ->(*,4ch,16,16)
        h = self.res4(h)  # ->(*,8ch,8,8)
        h = self.res5(h)  # ->(*,16ch,4,4)
        h = self.res6(h)  # ->(*,32ch,4,4)

        h = torch.sum((F.leaky_relu(h, 0.2)).view(batch, -1, 4 * 4),
                      dim=2)  # GlobalSumPool ->(*,16ch)
        outputs = self.fc(h)  # ->(*,1)

        if label is not None:
            embed = self.embedding(label)  # ->(*,16ch)
            outputs += torch.sum(embed * h, dim=1, keepdim=True)  # ->(*,1)

        return outputs
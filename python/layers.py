import torch.nn as nn
import torch
import torch.nn.functional as F
from sync_batchnorm import SynchronizedBatchNorm2d as SyncBN2d

def conv3x3(in_channel, out_channel):  # not change resolusion
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=1, padding=1, dilation=1, bias=False)


def conv1x1(in_channel, out_channel):  # not change resolution
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=1, padding=0, dilation=1, bias=False)


def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()

    elif classname.find('Batch') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.zero_()

    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()

    elif classname.find('Embedding') != -1:
        nn.init.orthogonal_(m.weight, gain=1)


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.theta = nn.utils.spectral_norm(
            conv1x1(channels, channels // 8)).apply(init_weight)
        self.phi = nn.utils.spectral_norm(
            conv1x1(channels, channels // 8)).apply(init_weight)
        self.g = nn.utils.spectral_norm(conv1x1(channels, channels // 2)).apply(
            init_weight)
        self.o = nn.utils.spectral_norm(conv1x1(channels // 2, channels)).apply(
            init_weight)
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, inputs):
        batch, c, h, w = inputs.size()
        theta = self.theta(inputs)  # ->(*,c/8,h,w)
        phi = F.max_pool2d(self.phi(inputs), [2, 2])  # ->(*,c/8,h/2,w/2)
        g = F.max_pool2d(self.g(inputs), [2, 2])  # ->(*,c/2,h/2,w/2)

        theta = theta.view(batch, self.channels // 8, -1)  # ->(*,c/8,h*w)
        phi = phi.view(batch, self.channels // 8, -1)  # ->(*,c/8,h*w/4)
        g = g.view(batch, self.channels // 2, -1)  # ->(*,c/2,h*w/4)

        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi),
                         -1)  # ->(*,h*w,h*w/4)
        o = self.o(
            torch.bmm(g, beta.transpose(1, 2)).view(batch, self.channels // 2,
                                                    h, w))  # ->(*,c,h,w)
        return self.gamma * o + inputs


class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, n_condition, cross_replica=False):
        super().__init__()
        if cross_replica:
            self.bn = SyncBN2d(in_channel, affine=False)
        else:
            self.bn = nn.BatchNorm2d(in_channel,
                                     affine=False)  # no learning parameters

        self.embed = nn.Linear(n_condition, in_channel * 2)

        nn.init.orthogonal_(self.embed.weight.data[:, :in_channel], gain=1)
        self.embed.weight.data[:, in_channel:].zero_()

    def forward(self, inputs, label):
        out = self.bn(inputs)
        embed = self.embed(label.float())
        gamma, beta = embed.chunk(2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = gamma * out + beta
        return out

class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


# BigGAN + leaky_relu
class ResBlock_G(nn.Module):
    def __init__(self, in_channel, out_channel, condition_dim, upsample=True, cross_replica=False):
        super().__init__()
        self.cbn1 = ConditionalNorm(in_channel, condition_dim, cross_replica=cross_replica)
        self.upsample = nn.Sequential()
        if upsample:
            self.upsample.add_module('upsample', nn.Upsample(scale_factor=2,
                                                             mode='nearest'))
        self.conv3x3_1 = nn.utils.spectral_norm(
            conv3x3(in_channel, out_channel)).apply(init_weight)
        self.cbn2 = ConditionalNorm(out_channel, condition_dim, cross_replica=cross_replica)
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

class MinibatchStdDev(nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    """
    def __init__(self):
        """
        derived class constructor
        """
        super(MinibatchStdDev, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape

        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)

        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)

        # return the computed values:
        return y
import torch.nn as nn
import torch
import torch.nn.functional as F

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
    def __init__(self, in_channel, n_condition):
        super().__init__()
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
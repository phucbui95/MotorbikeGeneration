import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm_notebook
import numpy as np
import math

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
    def __init__(self, n_feat, max_resolution,
                 codes_dim,
                 n_classes=0,
                 use_attention=False,
                 arch=None):
        super().__init__()
        self.codes_dim = codes_dim

        # construct residual blocks
        n_layers = int(np.log2(max_resolution) - 2)
        self.residual_blocks = nn.ModuleList([])
        last_block_dim = 0
        if arch is None:
            first_block_factor = 2 ** (n_layers)
        else:
            first_block_factor = arch[0]
        self.fc = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Linear(codes_dim,
                          first_block_factor * n_feat * 4 * 4)).apply(
                init_weight)
        )
        # print("first_block", first_block_factor)
        # print("n_layers ", n_layers)
        for i in range(n_layers):
            if arch is None:
                prev_factor = 2 ** (n_layers - i)
                curr_factor = 2 ** (n_layers - i - 1)
            else:
                prev_factor = arch[i]
                curr_factor = arch[i + 1]
            # print(f"block ({i}): {prev_factor}, {curr_factor}")
            block = ResBlock_G(prev_factor * n_feat, curr_factor * n_feat,
                               codes_dim + codes_dim, upsample=True)
            # add current block to the model class
            self.residual_blocks.add_module(f'res_block_{i}', block)
            if i == n_layers - 1:
                last_block_dim = curr_factor

        if use_attention:
            self.attn = Attention(2 * n_feat)

        # print("last_layer ", last_block_dim)
        self.to_rgb = nn.Sequential(
            # nn.BatchNorm2d(2*n_feat).apply(init_weight),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(conv3x3(last_block_dim * n_feat, 3)).apply(
                init_weight),
            nn.Tanh()
        )

        self.embedding = nn.Embedding(num_embeddings=n_classes,
                                      embedding_dim=self.codes_dim).apply(
            init_weight)

    def forward(self, z, label_ohe):
        '''
        z.shape = (*,Z_DIM)
        cd.shape = (*,n_classes)
        '''
        batch = z.size(0)
        z = z.squeeze()
        label_ohe = self.embedding(label_ohe)
        codes = torch.split(z, self.codes_dim, dim=1)

        x = self.fc(codes[0])  # ->(*,16ch*4*4)
        x = x.view(batch, -1, 4, 4)  # ->(*,16ch,4,4)
        # #print(f"feat={x.shape}")
        for block_index, block in enumerate(self.residual_blocks):
            # #print(f"block {block_index}")
            condition = torch.cat([codes[block_index + 1], label_ohe], dim=1)
            # #print(f"x.shape={x.shape}, condition.shape={condition.shape}")
            x = block(x, condition)
        x = self.to_rgb(x)
        return x


def test_generator():
    res = 128
    n_classes = 30
    g = Generator(
        n_feat=32,
        max_resolution=res,
        codes_dim=32,
        n_classes=n_classes,
        use_attention=False,
        arch=[16, 16, 8, 4, 2, 1]
    )

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Numer of parameters: ", count_parameters(g))
    print("{:=^90}".format('Model'))
    print(g)
    print("{:=^90}".format(''))

    inp = torch.randn((2, 32 * 8))
    ohe = torch.randn((2, n_classes))
    with torch.no_grad():
        output = g(inp, ohe)
    print("Output shape: ", output.shape)

    assert output.shape[2] == res


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
    def __init__(self, n_feat,
                 max_resolution,
                 n_classes=0,
                 use_dropout=None,
                 use_attention=False,
                 arch=None):
        super().__init__()
        self.max_resolution = max_resolution
        self.use_dropout = use_dropout
        self.res1 = ResBlock_D(3, n_feat, downsample=True)
        self.use_attention = use_attention
        if use_attention:
            self.attn = Attention(n_feat)

        self.residual_blocks = nn.ModuleList([])
        n_layers = int(np.log2(self.max_resolution)) - 2
        last_block_factor = 0

        for i in range(n_layers):
            is_last = (i == n_layers - 1)
            if arch is None:
                prev_factor = 2 ** (i)
                curr_factor = 2 ** (i + 1)
            else:
                prev_factor = arch[i]
                curr_factor = arch[i + 1]
            # print(f"block ({i}): {prev_factor}, {curr_factor}")
            block = ResBlock_D(prev_factor * n_feat, curr_factor * n_feat,
                               downsample=not is_last)
            self.residual_blocks.add_module(f"res_block_{i}", block)
            if is_last:
                last_block_factor = curr_factor

        if self.use_dropout is not None:
            self.dropout = nn.Dropout(self.use_dropout)

        self.fc = nn.utils.spectral_norm(
            nn.Linear(last_block_factor * n_feat, 1)).apply(
            init_weight)
        self.embedding = nn.Embedding(num_embeddings=n_classes,
                                      embedding_dim=last_block_factor * n_feat).apply(
            init_weight)

    def forward(self, inputs, label):
        batch = inputs.size(0)  # (*,3,128,128)
        h = self.res1(inputs)  # ->(*,ch,64,64)
        if self.use_attention:
            h = self.attn(h)

        for block_index, block in enumerate(self.residual_blocks):
            # print(f"block_{block_index}: h.shape={h.shape}")
            h = block(h)
        if self.use_dropout is not None:
            h = self.dropout(h)
        h = torch.sum((F.leaky_relu(h, 0.2)).view(batch, -1, 4 * 4),
                      dim=2)  # GlobalSumPool ->(*,16ch)
        outputs = self.fc(h)  # ->(*,1)

        if label is not None:
            embed = self.embedding(label)  # ->(*,16ch)
            # print(f"label.shape={label.shape}")
            # print(f"embedding_output= {embed.shape}")
            # print(f"h.shape={h.shape}")
            # print(f"output.shape={outputs.shape}")
            outputs += torch.sum(embed * h, dim=1, keepdim=True)  # ->(*,1)

        return outputs


def test_discriminator():
    res = 128
    n_classes = 30
    d = Discriminator(
        n_feat=32,
        max_resolution=res,
        n_classes=n_classes,
        use_attention=False,
        arch=[1, 2, 4, 8, 16, 16]
    )

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    inp = torch.randn((2, 3, res, res))
    cls = torch.from_numpy(np.random.randint(0, n_classes, (2,)))

    print("Numer of parameters: ", count_parameters(d))
    print("{:=^90}".format('Model'))
    print(d)
    # torchsummary.summary(d, [(3, res, res), (n_classes,)])
    print("{:=^90}".format(''))

    with torch.no_grad():
        output = d(inp, cls)
    print("Output shape: ", output.shape)


if __name__ == '__main__':
    test_generator()
    test_discriminator()
from dataloader import MotorbikeDataloader, sample_latent_vector, test_sample_latent_vector
from dataset import MotorbikeWithLabelsDataset, get_transforms
from gan_models import Generator, Discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import math
from functools import partial

import argparse


class GANTrainer:
    def __init__(self, opt, generator_fnc, discriminator_fnc):
        self.opt = opt

        self.netG = generator_fnc(opt)  # running parameters
        self.netD = discriminator_fnc(opt)
        self.netGE = generator_fnc(opt)  # exponential move average parameters
        self.netGE.load_state_dict(self.netG.state_dict())  # copy parameters

        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.opt.lr_D,
                                     betas=(self.opt.beta1, self.opt.beta2))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.opt.lr_G,
                                     betas=(self.opt.beta1, self.opt.beta2))

    def resume(self, file_name):
        checkpoint = torch.load(file_name)
        self.netD.load_state_dict(checkpoint['netD'])
        self.netG.load_state_dict(checkpoint['netG'])
        self.netGE.load_state_dict(checkpoint['netGE'])
        self.optimizerD.load_state_dict(checkpoint['optimizer_D'])
        self.optimizerG.load_state_dict(checkpoint['optimizer_G'])

        self.current_epoch = checkpoint['epoch']
        print("Loaded successfullly save models")

    def save(self, filename):
        current_state = {
            'netG': self.netG.state_dict(),
            'netD': self.netD.state_dict(),
            'netGE': self.netGE.state_dict(),
            'optimizer_D': self.optimizerD.state_dict(),
            'optimizer_G': self.optimizerG.state_dict(),
            'epoch': self.current_epoch
        }
        print(f"Saved model to {filename}")
        torch.save(current_state, filename)

    def ema_step(self, netG, netGE):
        # Update Generator Eval
        for param_G, param_GE in zip(netG.parameters(), netGE.parameters()):
            param_GE.data.mul_(self.opt.ema).add_(
                (1 - self.opt.ema) * param_G.data.cpu())
        for buffer_G, buffer_GE in zip(netG.buffers(), netGE.buffers()):
            buffer_GE.data.mul_(self.opt.ema).add_(
                (1 - self.opt.ema) * buffer_G.data.cpu())

    # Convenience utility to switch off requires_grad
    @staticmethod
    def toggle_grad(model, on_or_off):
        for param in model.parameters():
            param.requires_grad = on_or_off


class GANLoss:
    def __init__(self):
        self.name = "Hinge Loss"

    def discriminator_loss(self, real_samples, real_labels, discriminator,
                           generator, latent_sample_fnc):
        real_output = discriminator(real_samples, real_labels)
        real_loss = torch.mean(F.relu(1 - real_output))

        latent, fake_labels, fake_labels_ohe = latent_sample_fnc()
        fake_samples = generator(latent, fake_labels_ohe)
        fake_output = discriminator(fake_samples, fake_labels)

        fake_loss = torch.mean(F.relu(1 + fake_output))
        discriminator_loss = (real_loss + fake_loss) / 2.0
        return discriminator_loss

    def generator_loss(self, discriminator, generator, latent_sample_fnc):
        latent, fake_labels, fake_labels_ohe = latent_sample_fnc()
        fake_samples = generator(latent, fake_labels_ohe)
        fake_output = discriminator(fake_samples, fake_labels)

        generator_loss = torch.mean(F.relu(1 + fake_output))
        return generator_loss, latent


class Trainer(GANTrainer):
    def __init__(self, opt, generator_fnc, discriminator_fnc):
        super().__init__(opt, generator_fnc, discriminator_fnc)

        self.loss = GANLoss()

    def train_discriminator(self, data_loader, discriminator, generator):
        loss = 0
        for accumulative_index in range(self.opt.accumulative_steps):
            real_samples, real_labels = data_loader.next_batch()
            latent_sample_fnc = data_loader.get_latent_sample_fnc()
            # calculate loss
            discriminator_loss = self.loss.discriminator_loss(real_samples,
                                                              real_labels,
                                                              discriminator,
                                                              generator,
                                                              latent_sample_fnc)

            loss += discriminator_loss.item() / float(real_samples.size(0))
            discriminator_loss.backward()

        self.optimizerD.step()
        return loss

    def train_generator(self, data_loader, discriminator, generator):
        loss = 0
        for accumulative_index in range(self.opt.accumulative_steps):
            latent_sample_fnc = data_loader.get_latent_sample_fnc()
            generator_loss, latent = self.loss.generator_loss(discriminator, generator,
                                                      latent_sample_fnc)
            loss += generator_loss.item() / float(latent.size(0))
            generator_loss.backward()

        self.optimizerG.step()

    def move_to_device(self, device):
        self.netD.to(device)
        self.netG.to(device)
        self.netGE.to(device)

    def train_loop(self, data_loader, iteration=2, device=None):
        if device is None:
            device = torch.device(
                'cpu' if not torch.cuda.is_available() else 'cuda')

        fixed_noise, fixed_aux_labels, fixed_aux_labels_ohe = data_loader.latent_sample()

        generator = self.netGE
        running_generator = self.netG
        discriminator = self.netD

        generator.train()
        discriminator.train()
        # allow resume training

        for iter in tqdm(range(iteration)):
            self.toggle_grad(running_generator, False)
            self.toggle_grad(discriminator, True)
            self.train_discriminator(data_loader, discriminator,
                                     running_generator)

            self.toggle_grad(running_generator, True)
            self.toggle_grad(discriminator, False)
            self.train_generator(data_loader, discriminator, running_generator)
            self.ema_step(running_generator, running_generator)

            # Callback


def parse_arguments():
    args = argparse.ArgumentParser(description="Main training loop testing")
    args.add_argument('--path', type=str, required=True)
    args.add_argument('--label_path', type=str, required=True)
    args.add_argument('--image_size', type=int, default=128)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--workers', type=int, default=0)
    args.add_argument('--shuffle', action='store_true')

    # model configure
    args.add_argument('--n_classes', type=int, default=30)
    args.add_argument('--accumulative_steps', type=int, default=2)
    args.add_argument('--latent_size', type=int, default=120)

    args.add_argument('--feat_G', type=int, default=24)
    args.add_argument('--feat_D', type=int, default=24)

    # Optimizer hyperparameters
    args.add_argument('--beta1', type=float, default=0)
    args.add_argument('--beta2', type=float, default=0.999)

    args.add_argument('--lr_D', type=float, default=0.0004)
    args.add_argument('--lr_G', type=float, default=0.0004)

    args.add_argument('--ema', type=float, default=0.999)
    # args.add_argument()
    opt = args.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_arguments()

    # test_sample_latent_vector()
    base_tfs, additional_tfs = get_transforms(image_size=128)
    ds = MotorbikeWithLabelsDataset(opt.path, opt.label_path, base_tfs,
                                    additional_tfs, in_memory=False)
    dl = MotorbikeDataloader(opt, ds)
    class_dist = dl.dataset.get_class_distributions()
    print(class_dist)
    print(sum(class_dist))

    def get_generator_fnc(opt):
        return Generator(n_feat=opt.feat_G, codes_dim=10, n_classes=opt.n_classes)

    def get_discriminator_fnc(opt):
        return Discriminator(n_feat=opt.feat_D, n_classes=opt.n_classes)


    trainer = Trainer(opt, get_generator_fnc, get_discriminator_fnc)
    trainer.train_loop(dl, iteration=2)

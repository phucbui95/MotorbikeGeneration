import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dataloader import MotorbikeDataloader
from dataset import MotorbikeWithLabelsDataset, get_transforms
from gan_models import Generator, Discriminator
from tensorboardX import SummaryWriter
from tqdm import tqdm
from visualization import make_grid_image
from s3_client import S3Storage
import matplotlib.pyplot as plt

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
        print("Loaded successfullly save models")

    def save(self, filename):
        current_state = {
            'netG': self.netG.state_dict(),
            'netD': self.netD.state_dict(),
            'netGE': self.netGE.state_dict(),
            'optimizer_D': self.optimizerD.state_dict(),
            'optimizer_G': self.optimizerG.state_dict(),
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
        real_loss.backward(retain_graph=True)

        latent, fake_labels, fake_labels_ohe = latent_sample_fnc()
        fake_samples = generator(latent, fake_labels_ohe)
        fake_output = discriminator(fake_samples, fake_labels)
        fake_loss = torch.mean(F.relu(1 + fake_output))
        fake_loss.backward(retain_graph=True)

        discriminator_loss = (real_loss + fake_loss) / 2.0
        return discriminator_loss

    def generator_loss(self, discriminator, generator, latent_sample_fnc):
        latent, fake_labels, fake_labels_ohe = latent_sample_fnc()
        fake_samples = generator(latent, fake_labels_ohe)
        fake_output = discriminator(fake_samples, fake_labels)

        generator_loss = - torch.mean(fake_output)
        return generator_loss, latent


class Trainer(GANTrainer):
    def __init__(self, opt, generator_fnc, discriminator_fnc):
        super().__init__(opt, generator_fnc, discriminator_fnc)

        self.loss = GANLoss()
        self.opt.name = self.opt.name + datetime.now().strftime(
            '%Y-%m-%d_%H-%M-%S')
        # visualization
        if not os.path.exists(opt.tensorboard_dir):
            os.makedirs(opt.tensorboard_dir)
        board = SummaryWriter(
            log_dir=os.path.join(opt.tensorboard_dir, opt.name))
        self.board = board
        if opt.checkpoint_mode == 's3':
            self.s3storage = S3Storage(self.opt.name)

    def summary(self):
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("{:=^90}".format("Generator"))
        print("{: >30}:{: <60}".format("Number of parameters", count_parameters(self.netG)))
        print("{:=^90}".format("Discriminator"))
        print("{: >30}:{: <60}".format("Number of parameters",
                                       count_parameters(self.netD)))


    def __del__(self):
        self.board.close()
        try:
            del self.s3storage
        except:
            pass

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
            # discriminator_loss.backward()

        self.optimizerD.step()
        return loss

    def train_generator(self, data_loader, discriminator, generator):
        loss = 0
        for accumulative_index in range(self.opt.accumulative_steps):
            latent_sample_fnc = data_loader.get_latent_sample_fnc()
            generator_loss, latent = self.loss.generator_loss(discriminator,
                                                              generator,
                                                              latent_sample_fnc)
            loss += generator_loss.item() / float(latent.size(0))
            generator_loss.backward()

        self.optimizerG.step()
        return loss

    def generate_images(self, data_loader, num_samples):
        """
        Generate sample image for visualization while training
        :param data_loader:
        :param num_samples:
        :return:
        """
        latent_samplers = data_loader.get_latent_sample_fnc(
            batch_size=num_samples)
        noise, _, labels_ohe = latent_samplers()
        with torch.no_grad():
            generated_images = self.netG(noise, labels_ohe)
            generated_images = generated_images * 0.5 + 0.5
        # return generated_images
        sample_images = np.transpose(generated_images.cpu().numpy(),
                                     [0, 2, 3, 1])
        return [sample_images[i, :, :, :] for i in range(num_samples)]

    def train_loop(self, data_loader, iteration=2, device=None):
        if device is None:
            device = torch.device(
                'cpu' if not torch.cuda.is_available() else 'cuda')

        fixed_noise, fixed_aux_labels, fixed_aux_labels_ohe = data_loader.latent_sample()

        generator = self.netGE
        running_generator = self.netG.to(device)
        discriminator = self.netD.to(device)

        generator.train()
        discriminator.train()
        # allow resume training

        for iter in tqdm(range(iteration)):

            discriminator.zero_grad()
            self.toggle_grad(running_generator, False)
            self.toggle_grad(discriminator, True)
            D_running_loss = self.train_discriminator(data_loader,
                                                      discriminator,
                                                      running_generator)

            running_generator.zero_grad()
            self.toggle_grad(running_generator, True)
            self.toggle_grad(discriminator, False)
            G_running_loss = self.train_generator(data_loader,
                                                  discriminator,
                                                  running_generator)
            self.ema_step(running_generator, generator)

            # Callback
            if iter % self.opt.logging_steps == 0:
                print('[{:d}/{:d}] D_loss = {:.3f}, G_loss = {:.3f}'.format(
                    iter, iteration, D_running_loss, G_running_loss))
                self.board.add_scalar('D_loss', D_running_loss,
                                      global_step=iter)
                self.board.add_scalar('G_loss', G_running_loss,
                                      global_step=iter)
                generated_imgs = self.generate_images(data_loader, 12)
                sample_image = make_grid_image(generated_imgs, cols=4)
                # board_add_images(self.board, 'combie', generated_imgs, iter + 1)
                visuals = np.transpose(
                    (sample_image[..., ::-1] * 255).astype(np.uint8), [2, 0, 1])
                self.board.add_image(f'combie', visuals, global_step=iter)
                sample_dir = self.opt.sample_dir
                name = self.opt.name
                dir = os.path.join(sample_dir, name)
                if not os.path.exists(dir):
                    os.makedirs(dir)
                fname = f'{dir}/sample_{iter}.png'
                plt.imsave(fname, sample_image)

            if iter % self.opt.checkpoint_steps == 0:
                checkpoint_dir = os.path.join(self.opt.checkpoint_dir,
                                              self.opt.name)
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                filename = f'{checkpoint_dir}/model_{iter}.pth'
                self.save(filename)

                if self.opt.checkpoint_mode == 's3':
                    try:
                        self.s3storage.send_async(filename, 'checkpoint')
                    except:
                        print("Error while checkpoints were uploading to s3")
                        pass


def add_argments(arg_parser):
    arg_parser.add_argument('--name', type=str,
                            required=True)  # name of experiment

    arg_parser.add_argument('--path', type=str, required=True)
    arg_parser.add_argument('--label_path', type=str, required=True)
    arg_parser.add_argument('--image_size', type=int, default=128)
    arg_parser.add_argument('--batch_size', type=int, default=32)
    arg_parser.add_argument('--workers', type=int, default=0)
    arg_parser.add_argument('--shuffle', action='store_true')

    # model configure
    arg_parser.add_argument('--n_classes', type=int, default=30)
    arg_parser.add_argument('--accumulative_steps', type=int, default=2)
    arg_parser.add_argument('--latent_size', type=int, default=120)
    arg_parser.add_argument('--code_dim', type=int, default=20)

    arg_parser.add_argument('--feat_G', type=int, default=24)
    arg_parser.add_argument('--feat_D', type=int, default=24)


    # Optimizer hyperparameters
    arg_parser.add_argument('--iteration', type=int, default=100)
    arg_parser.add_argument('--beta1', type=float, default=0)
    arg_parser.add_argument('--beta2', type=float, default=0.999)

    arg_parser.add_argument('--lr_D', type=float, default=0.0004)
    arg_parser.add_argument('--lr_G', type=float, default=0.0004)

    arg_parser.add_argument('--ema', type=float, default=0.999)

    # Logging
    arg_parser.add_argument('--logging_steps', type=int, default=10)
    arg_parser.add_argument('--checkpoint_steps', type=int, default=500)
    arg_parser.add_argument('--sample_dir', type=str, default='sample')
    arg_parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    arg_parser.add_argument('--tensorboard_dir', type=str,
                            default='tensorboard_log')
    arg_parser.add_argument('--checkpoint_mode', type=str, default='local',
                            help='local|s3')

    # Checkpoint
    arg_parser.add_argument('--ckpt', type=str, default=None)

def parse_arguments():
    args = argparse.ArgumentParser(description="Main training loop testing")
    add_argments(args)
    # args.add_argument()
    opt = args.parse_args()
    return opt


def display_argments(opt):
    """Helper function for pretty printing arguments"""
    print("{:=^90}".format('Auguments'))
    for k, v in opt.__dict__.items():
        print("{: <20}:{: >80}".format(str(k),str(v)))

if __name__ == '__main__':
    opt = parse_arguments()
    display_argments(opt)
    # test_sample_latent_vector()
    base_tfs, additional_tfs = get_transforms(image_size=opt.image_size)
    ds = MotorbikeWithLabelsDataset(opt.path, opt.label_path, base_tfs,
                                    additional_tfs, in_memory=False)
    dl = MotorbikeDataloader(opt, ds)
    class_dist = dl.dataset.get_class_distributions()

    # print(class_dist)
    # print(sum(class_dist))
    def get_generator_fnc(opt):
        return Generator(n_feat=opt.feat_G,
                         max_resolution=opt.image_size,
                         codes_dim=opt.code_dim,
                         n_classes=opt.n_classes)


    def get_discriminator_fnc(opt):
        return Discriminator(n_feat=opt.feat_D, max_resolution=opt.image_size,
                             n_classes=opt.n_classes)

    trainer = Trainer(opt, get_generator_fnc, get_discriminator_fnc)
    trainer.summary()
    if opt.ckpt is not None and os.path.exists(opt.ckpt):
        print("{:=^90}".format(''))
        print(f"Loading from a checkpoint at {opt.ckpt}")
        print("{:=^90}".format(''))
        trainer.resume(opt.ckpt)
    trainer.train_loop(dl, iteration=opt.iteration)
    del trainer

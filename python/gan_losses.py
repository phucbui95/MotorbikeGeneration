""" Module implementing various loss functions """

import torch as th
import torch
import torch.nn.functional as F
import numpy as np

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (B, C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(2)
        w = img.size(3)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class CondHingeGAN:
    def __init__(self):
        self.name = "Hinge Loss"

    def discriminator_loss(self, real_samples, real_labels, discriminator,
                           generator, latent_sample_fnc):
        real_output = discriminator(real_samples, real_labels)
        real_loss = torch.mean(F.relu(1 - real_output))
        real_loss.backward(retain_graph=True)

        latent, fake_labels, fake_labels_ohe = latent_sample_fnc()
        fake_samples = generator(latent, fake_labels)
        fake_output = discriminator(fake_samples, fake_labels)
        fake_loss = torch.mean(F.relu(1 + fake_output))
        fake_loss.backward(retain_graph=True)

        discriminator_loss = (real_loss + fake_loss) / 2.0
        return discriminator_loss

    def generator_loss(self, real_samples, real_labels, discriminator,
                       generator, latent_sample_fnc):
        latent, fake_labels, fake_labels_ohe = latent_sample_fnc()
        fake_samples = generator(latent, fake_labels)
        fake_output = discriminator(fake_samples, fake_labels)

        generator_loss = - torch.mean(fake_output)
        return generator_loss, latent


class CondHingeGANWithCR:
    def __init__(self, tfs, gamma=10):
        self.name = "Hinge Loss"
        self.tfs = tfs
        self.gamma = gamma

    def _CR(self, real_samples, real_labels, discriminator):
        outputs, h0 = discriminator(real_samples, real_labels)
        real_samples_transformed = self.tfs(real_samples)
        _, h1 = discriminator(real_samples_transformed, real_labels)
        return torch.mean(torch.sqrt((h0 - h1).pow(2)))

    def discriminator_loss(self, real_samples, real_labels, discriminator,
                           generator, latent_sample_fnc):
        real_output, _ = discriminator(real_samples, real_labels)
        real_loss = torch.mean(F.relu(1 - real_output))
        real_loss.backward(retain_graph=True)

        latent, fake_labels, fake_labels_ohe = latent_sample_fnc()
        fake_samples = generator(latent, fake_labels_ohe)
        fake_output, _ = discriminator(fake_samples, fake_labels)
        fake_loss = torch.mean(F.relu(1 + fake_output))
        fake_loss.backward(retain_graph=True)

        cr_loss = self._CR(real_samples, real_labels, discriminator)
        discriminator_loss = (real_loss + fake_loss) / 2.0 + self.gamma * cr_loss
        return discriminator_loss

    def generator_loss(self, real_samples, real_labels, discriminator,
                       generator, latent_sample_fnc):
        latent, fake_labels, fake_labels_ohe = latent_sample_fnc()
        fake_samples = generator(latent, fake_labels_ohe)
        fake_output, _ = discriminator(fake_samples, fake_labels)

        generator_loss = - torch.mean(fake_output)
        return generator_loss, latent


class CondRelativisticAverageHingeGAN:

    def discriminator_loss(self, real_samples, real_labels, discriminator,
                           generator, latent_sample_fnc):
        # Obtain predictions
        latent, fake_labels, fake_labels_ohe = latent_sample_fnc()

        r_preds = discriminator(real_samples, real_labels)
        fake_samples = generator(latent, fake_labels_ohe)
        f_preds = discriminator(fake_samples, fake_labels)

        # difference between real and fake:
        r_f_diff = r_preds - th.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - th.mean(r_preds)

        # return the loss
        loss = (th.mean(th.nn.ReLU()(1 - r_f_diff))
                + th.mean(th.nn.ReLU()(1 + f_r_diff)))

        return loss

    def generator_loss(self, real_samples, real_labels, discriminator,
                           generator, latent_sample_fnc):
        # Obtain predictions
        r_preds = discriminator(real_samples, real_labels)
        latent, fake_labels, fake_labels_ohe = latent_sample_fnc()
        fake_samples = generator(latent, fake_labels_ohe)
        f_preds = discriminator(fake_samples, fake_labels)

        # difference between real and fake:
        r_f_diff = r_preds - th.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - th.mean(r_preds)

        # return the loss
        return (th.mean(th.nn.ReLU()(1 + r_f_diff))
                + th.mean(th.nn.ReLU()(1 - f_r_diff))), latent

def get_loss_function_by_name(name):
    if name == 'hinge':
        return CondHingeGAN()
    elif name == 'rahinge':
        return CondRelativisticAverageHingeGAN()
    elif name == 'crhinge':
        return CondHingeGANWithCR(tfs=Cutout(n_holes=5, length=4), gamma=2)
    else:
        raise Exception('Unknow loss')
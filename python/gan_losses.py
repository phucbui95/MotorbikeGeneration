""" Module implementing various loss functions """

import torch as th
import torch
import torch.nn.functional as F

class CondHingeGAN:
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

    def generator_loss(self, real_samples, real_labels, discriminator,
                       generator, latent_sample_fnc):
        latent, fake_labels, fake_labels_ohe = latent_sample_fnc()
        fake_samples = generator(latent, fake_labels_ohe)
        fake_output = discriminator(fake_samples, fake_labels)

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
    else:
        raise Exception('Unknow loss')
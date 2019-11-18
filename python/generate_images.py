import torch
import torch.nn as nn

from gan_models import Generator
from gan_trainer import parse_arguments, display_argments

import shutil
import os
import numpy as np
from torchvision.utils import save_image
from torchvision import transforms
import pandas as pd
from tqdm import tqdm


# class_distribution = [float(i) for i in class_distribution.split()]

def sample_latent_vector(class_distributions, latent_size, batch_size, device):
    """Util function to sample latent vectors from specified distribution"""
    noise = torch.randn(batch_size, latent_size, 1, 1, device=device)
    n_classes = len(class_distributions)
    aux_labels = np.random.choice(n_classes, batch_size, p=class_distributions)
    aux_labels_ohe = np.eye(n_classes)[aux_labels]
    aux_labels_ohe = torch.from_numpy(
        aux_labels_ohe[:, :, np.newaxis, np.newaxis])
    aux_labels_ohe = aux_labels_ohe.float().to(device, non_blocking=True)
    aux_labels = torch.from_numpy(aux_labels).to(device)
    return noise, aux_labels, aux_labels_ohe


def submission_generate_images(netG,
                               class_distribution,
                               n_images=10000,
                               device=None):
    im_batch_size = 50
    nz = 120
    if device is None:
        device = torch.device('cpu')

    netG.to(device)
    if not os.path.exists('outputs/output_images'):
        os.makedirs('outputs/output_images')

    output_i = 0
    pbar = tqdm(total=n_images)
    for i_batch in range(0, n_images, im_batch_size):
        gen_z, class_lbl, class_lbl_ohe = sample_latent_vector(
            class_distribution, nz, im_batch_size, device)
        gen_images = netG(gen_z, class_lbl_ohe)

        gen_images = gen_images.to(
            "cpu").clone().detach()  # shape=(*,3,h,w), torch.Tensor
        gen_images = gen_images * 0.5 + 0.5

        for i_image in range(gen_images.size(0)):
            out_path = os.path.join(f'outputs/output_images', f'{output_i}.png')
            out_img = (gen_images.numpy())[i_image, ::-1, :, :].copy()
            save_image(torch.tensor(out_img), out_path)
            output_i += 1
            pbar.update(output_i)
    pbar.close()
    shutil.make_archive(f'outputs/images', 'zip', f'outputs/output_images')


if __name__ == '__main__':
    opt = parse_arguments()
    display_argments(opt)

    G = Generator(n_feat=opt.feat_G,
                  max_resolution=opt.image_size,
                  codes_dim=opt.code_dim,
                  n_classes=opt.n_classes)

    if opt.ckpt is None:
        print("[ERROR] ckpt input required")
        exit(-1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(opt.ckpt, map_location=device)
    G.load_state_dict(ckpt['netGE'])

    df = pd.read_csv(opt.label_path)
    class_dist = (df['class'].value_counts() / len(df)).sort_index().values

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    submission_generate_images(G, class_dist, device=device)

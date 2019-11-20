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
            pbar.update(1)
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

    if os.path.exists(opt.label_path):
        df = pd.read_csv(opt.label_path)
        class_dist = (df['class'].value_counts() / len(df)).sort_index().values
    else:
        class_dist = [2.20957159e-02, 7.83481281e-02, 3.70513315e-02,
                      3.45426476e-02,
                      3.64724045e-02, 5.78927055e-03, 7.81551525e-03,
                      1.04206870e-02,
                      6.04978773e-02, 8.61636434e-02, 8.87688151e-03,
                      2.93323041e-02,
                      3.58934774e-02, 2.50868391e-03, 9.56194519e-02,
                      4.45773832e-02,
                      5.11385565e-03, 2.88498649e-02, 1.11153995e-01,
                      9.64878425e-05,
                      4.39984562e-02, 4.82439213e-04, 2.69201081e-02,
                      1.73678117e-03,
                      1.52354303e-01, 1.92975685e-04, 6.75414898e-04,
                      4.82439213e-03,
                      4.72790428e-03, 2.28676187e-02
                      ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    submission_generate_images(G, class_dist, device=device)

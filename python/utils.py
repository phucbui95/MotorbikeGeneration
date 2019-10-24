import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageOps, ImageEnhance
from torchvision import transforms
from torchvision.utils import save_image
import random


def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in
                                 range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

def generate_img(netG,fixed_noise,fixed_aux_labels=None):
    if fixed_aux_labels is not None:
        gen_image = netG(fixed_noise,fixed_aux_labels).to('cpu').clone().detach().squeeze(0)
    else:
        gen_image = netG(fixed_noise).to('cpu').clone().detach().squeeze(0)
    #denormalize
    gen_image = gen_image*0.5 + 0.5
    gen_image_numpy = gen_image.numpy().transpose(0,2,3,1)
    return gen_image_numpy

def show_generate_imgs(netG, fixed_noise, max_image=4, fixed_aux_labels=None):
    gen_images_numpy = generate_img(netG,fixed_noise,fixed_aux_labels)

    fig = plt.figure(figsize=(25, 16))
    # display 10 images from each class
    for i, img in enumerate(gen_images_numpy[:max_image]):
        ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])
        plt.imshow(img)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_weights(model, filename, verbose=1):
    if verbose:
        print(f'-> Saving weights to {filename}')
    torch.save(model.state_dict(), filename)


def load_model_weights(model, filename, verbose=1):
    if verbose:
        print(f'-> Loading weights from {filename}')
    model.load_state_dict(torch.load(filename, map_location='cpu'))
    return model


def autocontrast(img, cutoff=1):  # cutoff[%]
    if np.random.rand() < 0.5:
        img = ImageOps.autocontrast(img, cutoff)
    return img


def sharpen(img, magnitude=1):
    factor = np.random.uniform(1.0 - magnitude, 1.0 + magnitude)
    img = ImageEnhance.Sharpness(img).enhance(factor)
    return img


# truncation_trick
def submission_generate_images(netG, n_images=10000, truncated=None,
                               device=None, shapen=None):
    im_batch_size = 50
    nz = 120

    if device:
        device = torch.device('cpu')

    if not os.path.exists('outputs/output_images'):
        os.mkdir('outputs/output_images')

    for i_batch in range(0, n_images, im_batch_size):
        if truncated is not None:
            flag = True
            while flag:
                z = np.random.randn(100 * im_batch_size * nz)
                z = z[np.where(abs(z) < truncated)]
                if len(z) >= im_batch_size * nz:
                    flag = False

            gen_z = torch.from_numpy(z[:im_batch_size * nz]).view(im_batch_size,
                                                                  nz, 1, 1)
            gen_z = gen_z.float().to(device)
        else:
            gen_z = torch.randn(im_batch_size, nz, 1, 1, device=device)
        #         gen_z = gen_z / gen_z.norm(dim=1, keepdim=True)
        gen_images = netG(gen_z)
        gen_images = gen_images.to(
            "cpu").clone().detach()  # shape=(*,3,h,w), torch.Tensor
        # denormalize
        gen_images = gen_images * 0.5 + 0.5

        for i_image in range(gen_images.size(0)):
            if shapen is not None:
                img = transforms.ToPILImage()(gen_images[i_image])
                img = sharpen(img, magnitude=shapen)
                img = transforms.ToTensor()(img)
                save_image(img,
                           os.path.join(f'outputs/output_images',
                                        f'image_{i_batch+i_image:05d}.png'))
            else:
                save_image(gen_images[i_image, :, :, :],
                           os.path.join(f'outputs/output_images',
                                        f'image_{i_batch+i_image:05d}.png'))
    shutil.make_archive(f'outputs/images', 'zip', f'outputs/output_images')


def seed_everything():
    # random seeds
    seed = 2019
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

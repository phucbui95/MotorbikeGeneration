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
import cv2
import tensorflow as tf


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


def post_preprocessing():
    from glob import glob
    img_paths = glob('outputs/intermediate_images/*.png')

    w, h = 128, 128

    img_np = np.empty((len(img_paths), w, h, 3), dtype=np.uint8)
    for idx, path in enumerate(img_paths):
        img_arr = cv2.imread(path)
        img_arr = cv2.resize(img_arr, (w, h), cv2.INTER_BITS)
        img_arr = img_arr[..., ::-1]
        img_arr = np.array(img_arr)
        img_np[idx] = img_arr

    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    PATH_TO_MODEL = './client/motorbike_classification_inception_net_128_v4_e36.pb'

    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        input_tensor = graph.get_tensor_by_name('input_1:0')
        output_tensor = graph.get_tensor_by_name('activation_95/Sigmoid:0')
        embedding_tensor = graph.get_tensor_by_name(
            'global_average_pooling2d_1/Mean:0')

    indicates = list(range(len(img_np)))
    batch_size = 32
    list_index = [indicates[i:i + batch_size] for i in
                  range(0, len(indicates), batch_size)]

    score_list = []

    for batch_index in list_index:
        img_expanded = img_np[batch_index] / 255.0

        with graph.as_default():
            scores = sess.run([output_tensor],
                              feed_dict={input_tensor: img_expanded})
        score_list.append(scores[0])
    score = pd.DataFrame({'path': img_paths,
                          'score': np.concatenate(score_list, axis=0).reshape(
                              -1)})
    high_quality = score[score['score'] >= score['score'].quantile(0.125)]
    img_path = high_quality['path'].values
    np.random.shuffle(img_path)
    img_path = img_path[:10000]

    output_dir = 'outputs/output_images'
    try:
        os.makedirs(output_dir)
    except:
        pass

    for i in range(len(img_path)):
        fpath = img_path[i]
        im = cv2.imread(fpath)
        cv2.imwrite(f'{output_dir}/{i}.png', im)
    shutil.make_archive(f'outputs/images', 'zip', output_dir)


def submission_generate_images(netG,
                               class_distribution,
                               n_images=15000,
                               device=None, nz=120):
    im_batch_size = 50
    if device is None:
        device = torch.device('cpu')

    netG.to(device)
    if not os.path.exists('outputs/intermediate_images'):
        os.makedirs('outputs/intermediate_images')

    output_i = 0
    pbar = tqdm(total=n_images)
    for i_batch in range(0, n_images, im_batch_size):
        gen_z, class_lbl, class_lbl_ohe = sample_latent_vector(
            class_distribution, nz, im_batch_size, device)
        gen_images = netG(gen_z, class_lbl)

        gen_images = gen_images.to(
            "cpu").clone().detach()  # shape=(*,3,h,w), torch.Tensor
        gen_images = gen_images * 0.5 + 0.5

        for i_image in range(gen_images.size(0)):
            out_path = os.path.join(f'outputs/intermediate_images', f'{output_i}.png')
            out_img = (gen_images.numpy())[i_image, :, :, :].copy()
            save_image(torch.tensor(out_img), out_path)
            output_i += 1
            pbar.update(1)
    pbar.close()
    post_preprocessing()


if __name__ == '__main__':
    opt = parse_arguments()
    display_argments(opt)

    if opt.use_dropout is None or opt.use_dropout < 0:
        opt.use_dropout = None

    opt.use_attention = opt.use_attention == '1'
    opt.cross_replica = opt.cross_replica == '1'

    arch = [16, 16, 8, 4, 2, 1]

    G = Generator(n_feat=opt.feat_G,
                  max_resolution=opt.image_size,
                  codes_dim=opt.code_dim,
                  n_classes=opt.n_classes,
                  arch=arch,
                  use_attention=opt.use_attention,
                  cross_replica=opt.cross_replica,
                  rgb_bn=False)

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
    submission_generate_images(G, class_dist, nz=opt.latent_size, device=device)

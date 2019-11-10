import argparse
import configparser
from configparser import ExtendedInterpolation

class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            config = configparser.ConfigParser(interpolation=ExtendedInterpolation())
            config.read(f.name)
        # print(config['DEFAULTS'])
        for k, v in config['DEFAULT'].items():
            setattr(namespace, k, v)

def set_arguments(parser):
    parser.add_argument('--path', required=False, default='', type=str, help='path of specified dataset')
    parser.add_argument(
        '--phase',
        type=int,
        default=600_000,
        help='number of samples used for each training phases',
    )
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    parser.add_argument('--max_size', default=1024, type=int, help='max image size')
    parser.add_argument('--batch_size', default=16, type=int, )
    parser.add_argument(
        '--ckpt', default=None, type=str, help='load from previous checkpoints'
    )
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )
    parser.add_argument(
        '--mixing', action='store_true', help='use mixing regularization'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='wgan-gp',
        choices=['wgan-gp', 'r1'],
        help='class of gan loss',
    )

def get_parser():
    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
    set_arguments(parser)
    # args = parser.parse_args()
    return parser

def get_parser_from_file():
    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
    set_arguments(parser)
    parser.add_argument('--file', type=open, action=LoadFromFile)
    return parser

if __name__ == '__main__':
    parser = get_parser_from_file()
    # other arguments
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    args = parser.parse_args()

    print(args)
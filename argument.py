
import argparse

#some args

def add_model_config_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument('--model-config', type=str, 
                       help='model configuration file') 
    return parser

def add_training_args(parser: argparse.ArgumentParser):
    """Training arguments."""

    group = parser.add_argument_group('train', 'training configurations')
    group.add_argument('--batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='gradient clipping')
    group.add_argument('--train-iters', type=int, default=1000000,
                       help='total number of iterations to train over all training runs')
    group.add_argument('--max-length', type=int, default=512,
                       help='max length of input')
    group.add_argument('--start-step', type=int, default=0,
                       help='step to start or continue training')
    group.add_argument('--seed', type=int, default=1234,
                       help='random seed for reproducibility')
    group.add_argument('--epochs', type=int, default=1,
                       help='total number of epochs to train over all training runs')
    # Learning rate.
    group.add_argument('--lr', type=float, default=1.0e-4,
                       help='initial learning rate')
    group.add_argument('--weight-decay', type=float, default=1.0e-2,
                       help='weight-decay')
    group.add_argument('--offloading', action='store_true', help='offloading_enable')
    group.add_argument('--fp16', action='store_true', help='offloading_enable')
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_model_config_args(parser)
    parser = add_training_args(parser)
    args = parser.parse_args()
    return args

import argparse
import sys

from utils import log

parser = argparse.ArgumentParser()

parser.add_argument('--d', type=int, default=3,
                    help='the distance of the original code, one of the labels of code')
parser.add_argument('--k', type=int, default=1,
        help='the number of logical qubits of the code, one of the labels of code, default: %(default)d')
parser.add_argument('--c_type', type=str, default='sur',
                    help='the code type of the original code, one of the labels of code, default: %(default)s')
parser.add_argument('--p', type=float, default=0.1,
                    help='deplorazed model error rate')
parser.add_argument('--trnsz', type=int, default=10000000,
                    help='the size of training set')
parser.add_argument('--testsz', type=int, default=10000,
                    help='the size of training set')
parser.add_argument('--epoch', type=int, default=1,
                    help='epoch')
parser.add_argument('--gpu', type=int, default=0,
                    help='gpu id')
parser.add_argument('--seed', type=int, default=0,
                    help="random seed for initialization")
parser.add_argument("--eval_every", default=100, type=int,
                    help="Run prediction on validation set every so many steps."
                         "Will always run one evaluation at the end of training.")
parser.add_argument("--train_batch_size", default=512, type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size", default=64, type=int,
                    help="Total batch size for eval.")
parser.add_argument("--zip", default=0, type=int,
                    help="whether to zip data")
parser.add_argument("--limit", default=10000000, type=int,
                    help="repeat num limit(only for zip)")
parser.add_argument("--poolsz", default=10000000, type=int,
                    help="train sample pool size before zip")
parser.add_argument('--sym', type=str, nargs='+',
                    help='a list of strings')

parser.add_argument('--nn', type=str, default='fnn', choices=['fnn', 'cnn', 'rnn'],
                    help='type of network')

parser.add_argument('--work', type=int, default=1,
                    help='num of work')

args = parser.parse_args()

if args.sym == ['rf']:
    args.sym = ['rf:0', 'rf:1', 'rf:2', 'rf:3']
elif args.sym == ['rt']:
    args.sym = ['rt:0', 'rt:1', 'rt:2']

log(f"args:\n{args}")

import argparse

parser = argparse.ArgumentParser(description='vanila VAE')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='alpha div parameter (default: 1.0)')
parser.add_argument('--beta', type=float, default=1.0,
                    help='div weight parameter (default: 1.0)')
parser.add_argument('--nu', type=float, default=0.0,
                    help='gamma div parameter (default: 0)')
parser.add_argument('--dataset', type=str, default="mnist",
                    help='Dataset name')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--seed', type=int, default=999,
                    help='set seed number (default: 999)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--zdim',  type=int, default=32,
                    help='the z size for training (default: 512)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--frac', type=float, default=0.5,
                    help='fraction of noisy dataset')
parser.add_argument('--no_cuda', action='store_true',
                    help='enables CUDA training')
parser.add_argument('--recon_sigma', type=float, default=1.0,
                    help='sigma value for covariance matrix for reconstruct t-dist')
parser.add_argument('--gpu_id', type=str, default='0',
                    help='gpu id')
parser.add_argument('--tsne', type=int, default=1,
                    help='make tsne plot (1:true, otherwise:false')
parser.add_argument('--method', type=int, default=1,
                    help='choice one of method dealing with 2nd loss term')


# parser.add_argument('-s', '--save', action='store_true', default=True,
#                     help='save model every epoch')
# parser.add_argument('-l', '--load', action='store_true',
#                     help='load model at the begining')
# parser.add_argument('--model_dir', type=str, default='',
#                     help='model storing path')
args = parser.parse_args()

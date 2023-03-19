import argparse

parser = argparse.ArgumentParser(description='gammaAE')
parser.add_argument('--dataset', type=str, default="mnist",
                    help='Dataset name')
parser.add_argument('--nu', type=float, default=0.0,
                    help='gamma div parameter (default: 0)')
parser.add_argument('--beta', type=float, default=0.0,
                    help='rvae div parameter (default: 0, no-use)')
parser.add_argument('--flat', type=str, default='y',
                    help='use gamma-pow regularizer')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--seed', type=int, default=2023,
                    help='set seed number (default: 2023)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--zdim',  type=int, default=32,
                    help='latent_zdim for training (default: 32)')
parser.add_argument('--lr', type=float, default=5e-5, 
                    help='learning rate')
parser.add_argument('--train_frac', type=float, default=0,
                    help='fraction ratio of noisy data in trainset')
parser.add_argument('--test_frac', type=float, default=0,
                    help='fraction ratio of noisy data in testset')
parser.add_argument('--reg_weight', type=float, default=1.0,
                    help='constant weight for regularizer (default: 1.0)')
parser.add_argument('--no_cuda', action='store_true',
                    help='enables CUDA training')
parser.add_argument('--recon_sigma', type=float, default=1,
                    help='sigma value in reconstruction term')
parser.add_argument('--gpu_id', type=str, default='0',
                    help='gpu id')
parser.add_argument('--tsne', type=int, default=0,
                    help='make tsne plot (1:true, otherwise:false')
args = parser.parse_args()

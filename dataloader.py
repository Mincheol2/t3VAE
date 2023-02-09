import argument
import torch
from torch.utils.data import DataLoader
#import torchvision
#import torchvision.transforms as transforms
from mnistc_dataset import *
import numpy as np
## Load trainset, testset and trainloader, testloader ###
# transform.Totensor() is used to normalize mnist data. Range : [0, 255] -> [0,1]
args = argument.args
MNISTC = MNISTC_Dataset()
trainset, testset = MNISTC.get_dataset('identity')
                              
# Load MNIST-C dataset
if args.dataset != "mnist":
    train_N = 60000 # Total : 60000
    test_N = 10000
    noise_trainset, noise_testset = MNISTC.get_dataset(args.dataset)

    def make_masking(N,frac):

        indice = np.arange(0,N)
        mask = np.zeros(N,dtype=bool)
        rand_indice = np.random.choice(N, int(frac*N))
        mask[rand_indice] = True
        
        return indice[mask], indice[~mask]

    I1, I2 = make_masking(train_N,args.train_frac)
    trainset = torch.utils.data.Subset(trainset, indices=I1)
    noise_trainset = torch.utils.data.Subset(noise_trainset, indices=I2)
    
    i1, i2 = make_masking(test_N,args.test_frac)
    testset = torch.utils.data.Subset(testset, indices=i1)
    noise_testset = torch.utils.data.Subset(noise_testset, indices=i2)
   
    # Train with MNISTC, and reconstruct MNIST data
    
    trainset = torch.utils.data.ConcatDataset([trainset, noise_trainset])
    testset = torch.utils.data.ConcatDataset([testset, noise_testset])



trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)

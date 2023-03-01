import argument
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnistc_dataset import *
import numpy as np
## Load trainset, testset and trainloader, testloader ###
# transform.Totensor() is used to normalize mnist data. Range : [0, 255] -> [0,1]

class MYTensorDataset(torch.utils.data.Dataset):
    def __init__(self, *tensors) -> None:
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

class Noisy_Dataset():
    def __init__(self, path, train=True, transform=None):
        self.path = "./" + path 
        
    def get_dataset(self, noise_name):
        try:
            curr_path = self.path + noise_name
            train_img = np.load(curr_path +'/train_images.npy')
            train_class = np.load(curr_path +'/train_labels.npy')
            test_img = np.load(curr_path + '/test_images.npy')
            test_class = np.load(curr_path + '/test_labels.npy')
        except:
            raise Exception("The dataset is not available.")
        
        
        # Transpose numpy array: [B, H, W, C] --> tensor [B, C, H, W]
        # And, normalize mnist pixel : range [0,255] -> [0,1]
        train_B = train_img.shape[0]
        test_B = test_img.shape[0]
        train_dataset = torch.zeros((train_B,1,28,28))
        test_dataset = torch.zeros((test_B,1,28,28))
        toTensor = transforms.ToTensor()
        for i in range(train_B):
            x = toTensor(train_img[i])
            train_dataset[i] = x
        for j in range(test_B):
            y = toTensor(test_img[j])
            test_dataset[j] = y
        
        # remove rawdata from our memory.
        del train_img
        del test_img

        train_data = MYTensorDataset(train_dataset, train_class)
        test_data = MYTensorDataset(test_dataset, test_class)
        return (train_data,test_data)

def make_masking(N,frac):
    indice = np.arange(0,N)
    mask = np.zeros(N,dtype=bool)
    rand_indice = np.random.choice(N, int(frac*N))
    mask[rand_indice] = True
    
    return indice[mask], indice[~mask]

def generate_dataloader(trainset,testset,dataset_name):
    train_N = 60000
    test_N = 10000
    args = argument.args
    noise_dataset = Noisy_Dataset(path=dataset_name)
    noise_trainset, noise_testset = noise_dataset.get_dataset(dataset_name)

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

    return trainloader, testloader

def load_mnist_dataset(dataset_name):
    args = argument.args
    
    transform = transforms.Compose([transforms.ToTensor(),
                               ])

    trainset = datasets.MNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    testset = datasets.MNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)
    
    ## Original data##
    if dataset_name == "mnist_default":
        return trainloader, testloader

    ## mix contamination data ##
    else:
        return generate_dataloader(trainset,testset,dataset_name)


def load_fashion_dataset(dataset_name):
    args = argument.args
    transform = transforms.Compose([transforms.ToTensor(),
                               ])
    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    if dataset_name == "fashion_default":
        return trainloader, testloader

    ## mix contamination data ##
    else:
        return generate_dataloader(trainset,testset,dataset_name)


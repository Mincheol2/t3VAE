import argument
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
## Load trainset, testset and trainloader, testloader ###
# transform.Totensor() is used to normalize mnist data. Range : [0, 255] -> [0,1]

'''
    return
    ((np_Etrainset, np_Etrainclass), (np_Etestset, np_Etestclass)) -> numpy arrays
'''
def get_noise_dataset(path):
    curr_path = "./datasets/" + path
    try:
        train_img = np.load(curr_path + '/train_images.npy')
        train_class = np.load(curr_path + '/train_labels.npy')
        test_img = np.load(curr_path + '/test_images.npy')
        test_class = np.load(curr_path + '/test_labels.npy')
    except:
        raise Exception("The dataset is not available.")

    # classify noise dataset label to negative.
    train_class = - train_class
    test_class = - test_class


    train_data = (train_img, train_class)
    test_data = (test_img, test_class)
    return(train_data,test_data)
    

'''
    split indices for concatenating two datasets

    Input
    N : total number of the concatenated dataset
    frac : fraction ratio of noise dataset

    return
    I1 : noise dataset indices , I2 : original dataset indices
'''
def make_masking(N,frac):
    indice = np.arange(0,N)
    mask = np.zeros(N,dtype=bool)
    rand_indice = np.random.choice(N, int(frac*N),replace=False)
    mask[rand_indice] = True
    return indice[mask], indice[~mask]


def transform_np_to_tensor(train_img,test_img):
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

    return (train_dataset, test_dataset)
'''
    Input
    origin_dataset, noise_dataset :
    each argument is structured as
    (np_trainset, np_trainclass), (np_testset, np_testclass))
'''
def generate_dataloader(origin_dataset, noise_dataset):
    train_N = 60000
    test_N = 10000
    args = argument.args
    
    trainset, trainclass = origin_dataset[0]
    testset, testclass = origin_dataset[1]
    noise_trainset, noise_trainclass = noise_dataset[0]
    noise_testset, noise_testclass = noise_dataset[1]


    trainset_tensor, testset_tensor = transform_np_to_tensor(trainset,testset)
    noise_trainset_tensor, noise_testset_tensor = transform_np_to_tensor(noise_trainset,noise_testset)
    sample_N1, sample_N2 = 24, 8
    sample_imgs = torch.cat((testset_tensor[:sample_N1], noise_testset_tensor[:sample_N2]),0)
    
    I1, _ = make_masking(train_N,args.train_frac)
    i1, _ = make_masking(test_N,args.test_frac)
    trainset_tensor[I1] = noise_trainset_tensor[I1]
    testset_tensor[i1] = noise_testset_tensor[i1]

    trainclass[I1] = noise_trainclass[I1]
    testclass[i1] = noise_testclass[i1]

    trainclass = torch.tensor(trainclass)
    testclass = torch.tensor(testclass)
    # Train with MNISTC, and reconstruct MNIST data
    trainset = torch.utils.data.TensorDataset(trainset_tensor, trainclass)
    testset = torch.utils.data.TensorDataset(testset_tensor, testclass)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    return trainloader, testloader,sample_imgs



def load_mnist_dataset(dataset_name):
    args = argument.args
    transform = transforms.Compose([transforms.ToTensor(),
                                   ])
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    ## Original data##
    if dataset_name == "mnist":
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)
        
        for images, _ in trainloader:
            sample_imgs = images
            break
        return trainloader, testloader, sample_imgs

    ## Mix with contamination data ##
    else:
        np_trainset = (trainset.data.numpy(), trainset.targets.numpy())
        np_testset = (testset.data.numpy(), testset.targets.numpy())
        origin_dataset = (np_trainset, np_testset)
        if dataset_name == "emnist":
            # Since the letters in the EMNIST data are flipped, we need to rotate and hflip.
            # Also, we use unified numpy array form : [B, H, W, C]
            Etrainset = datasets.EMNIST('~/.pytorch/EMNIST_data/', download=True, split='letters',train=True)
            Etestset = datasets.EMNIST('~/.pytorch/EMNIST_data/', download=True, split='letters', train=False)
            
            np_Etrainset = np.expand_dims(np.rot90(np.flip(Etrainset.data.numpy(),axis=1),3, axes=(1,2)),3)
            np_Etestset = np.expand_dims(np.rot90(np.flip(Etrainset.data.numpy(),axis=1),3, axes=(1,2)),3)

            # convert positive emnist label to negative
            np_Etrainclass = np.array(-Etrainset.targets)
            np_Etestclass = np.array(-Etestset.targets)
            noise_dataset = ((np_Etrainset, np_Etrainclass), (np_Etestset, np_Etestclass))
        else:
            noise_dataset = get_noise_dataset(path=dataset_name)
        
        return generate_dataloader(origin_dataset,noise_dataset)

    

def load_fashion_dataset(dataset_name):
    args = argument.args
    transform = transforms.Compose([transforms.ToTensor(),
                               ])
    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)

    testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)

    ## Original data##
    if dataset_name == "fashion":
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)
        
        for images, _ in trainloader:
            sample_imgs = images
            break
        return trainloader, testloader, sample_imgs

    ## Mix with contamination data ##
    else:
        np_trainset = (trainset.data.numpy(), trainset.targets.numpy())
        np_testset = (testset.data.numpy(), testset.targets.numpy())
        origin_dataset = (np_trainset, np_testset)
        noise_dataset = get_noise_dataset(path=dataset_name)
        
        return generate_dataloader(origin_dataset,noise_dataset)



def select_dataloader():
    args = argument.args
    if 'mnist' in args.dataset:
        trainloader, testloader, tensorboard_imgs = load_mnist_dataset(args.dataset)
    elif 'fashion' in args.dataset:
        trainloader, testloader, tensorboard_imgs = load_fashion_dataset(args.dataset)

    return trainloader, testloader, tensorboard_imgs


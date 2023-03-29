import argument
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

## Load trainset, testset and trainloader, testloader ###
# transform.Totensor() is used to normalize mnist data. Range : [0, 255] -> [0,1]

class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        return img_transformed, 0 # We don't use any label now.

   




'''
    return
    ((np_Etrainset, np_Etrainclass), (np_Etestset, np_Etestclass)) -> numpy arrays
'''



def get_noise_dataset(path):
    curr_path = "/data_intern/" + path # shared data path.
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
    # And normalize pixels : range [0,255] -> [0,1]
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
    sample_N1, sample_N2 = 16,16
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

    transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4,transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=4,transform=transform)

    return trainloader, testloader,sample_imgs

class load_dataset():

    def __init__(self):
        self.args = argument.args

        # data range : -1 ~ 1
        self.transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((32, 32))
        ]
        )

    def load_mnist_dataset(self, dataset_name):
        self.transform = transforms.Compose([transforms.ToTensor(),
        transforms.Resize((32, 32))])
        trainset = datasets.MNIST('/data_intern/MNIST/', download=True, train=True, transform=self.transform)
        testset = datasets.MNIST('/data_intern/MNIST/', download=True, train=False, transform=self.transform)
        ## Original data##
        if dataset_name == "mnist":
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)
            testloader = torch.utils.data.DataLoader(testset, batch_size=self.args.batch_size, shuffle=True)
            
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
                Etrainset = datasets.EMNIST('/data_intern/EMNIST/', download=True, split='letters',train=True, transform=self.transform)
                Etestset = datasets.EMNIST('/data_intern/EMNIST/', download=True, split='letters', train=False, transform=self.transform)
                
                np_Etrainset = np.expand_dims(np.rot90(np.flip(Etrainset.data.numpy(),axis=1),3, axes=(1,2)),3)
                np_Etestset = np.expand_dims(np.rot90(np.flip(Etrainset.data.numpy(),axis=1),3, axes=(1,2)),3)

                # convert positive emnist label to negative
                np_Etrainclass = np.array(-Etrainset.targets)
                np_Etestclass = np.array(-Etestset.targets)
                noise_dataset = ((np_Etrainset, np_Etrainclass), (np_Etestset, np_Etestclass))
            else:
                noise_dataset = get_noise_dataset(path=dataset_name)
            
            return generate_dataloader(origin_dataset,noise_dataset)
    
    def load_fashion_dataset(self, dataset_name):
        trainset = datasets.FashionMNIST('/data_intern/FashionMNIST/', download=True, train=True, transform=self.transform)

        testset = datasets.FashionMNIST('/data_intern/FashionMNIST/', download=True, train=False, transform=self.transform)

        ## Original data##
        if dataset_name == "fashion":
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)
            testloader = torch.utils.data.DataLoader(testset, batch_size=self.args.batch_size, shuffle=True)
            
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

    def load_cifar_dataset(self, dataset_name):
        trainset = datasets.CIFAR10('/data_intern/CIFAR10/', download=True, train=True, transform=self.transform)
        testset = datasets.CIFAR10('/data_intern/CIFAR10/', download=True, train=False, transform=self.transform)

        ## Original data##
        if dataset_name == "cifar10":
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)
            testloader = torch.utils.data.DataLoader(testset, batch_size=self.args.batch_size, shuffle=True)
            
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

    def load_celeb_dataset(self, dataset_name):

        self.transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        )
        train_img_list = []
        test_img_list = []
        celeb_total_nb = 202530

        # We manually split the train/test set.
        # Train set : 60000 / Test set : 10000
        # For comparison, we select indices previously.

        # indices = np.random.choice(celeb_total_nb, 70000,replace=False)
        # train_indices = indices[:60000]
        # test_indices = indices[60000:]

        train_indices = range(1,60001) # 1~60000
        test_indices = range(60001, 70001) # 60001~70000
        if dataset_name == "celeb_crop64":
            for idx in train_indices:
                img_path = f"/data_intern/celeba_crop64/{idx:06d}.jpg"
                train_img_list.append(img_path)
            for idx in test_indices:
                img_path = f"/data_intern/celeba_crop64/{idx:06d}.jpg"
                test_img_list.append(img_path)
        elif dataset_name == "celeb_crop128":
            for idx in train_indices:
                img_path = f"/data_intern/celeba_crop128/{idx:06d}.jpg"
                train_img_list.append(img_path)
            for idx in test_indices:
                img_path = f"/data_intern/celeba_crop128/{idx:06d}.jpg"
                test_img_list.append(img_path)
        else:
            raise Exception("Use proper size.")


        trainset = Custom_Dataset(file_list=train_img_list,
                            transform=self.transform)

        testset = Custom_Dataset(file_list=test_img_list,
                            transform=self.transform)
        ## Original data ##
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.args.batch_size, shuffle=True)
        
        for images, _ in trainloader:
            sample_imgs = images
            break
        return trainloader, testloader, sample_imgs

        # ## Mixed with contamination data (TODO) ##
        # else:
        #     np_trainset = (trainset.data.numpy(), trainset.targets.numpy())
        #     np_testset = (testset.data.numpy(), testset.targets.numpy())
        #     origin_dataset = (np_trainset, np_testset)
        #     noise_dataset = get_noise_dataset(path=dataset_name)
            
        #     return generate_dataloader(origin_dataset,noise_dataset)

    def select_dataloader(self):
        if 'mnist' in self.args.dataset:
            trainloader, testloader, tensorboard_imgs = self.load_mnist_dataset(self.args.dataset)
        elif 'fashion' in self.args.dataset:
            trainloader, testloader, tensorboard_imgs = self.load_fashion_dataset(self.args.dataset)
        elif 'cifar' in self.args.dataset:
            trainloader, testloader, tensorboard_imgs = self.load_cifar_dataset(self.args.dataset)
        elif 'celeb' in self.args.dataset:
            trainloader, testloader, tensorboard_imgs = self.load_celeb_dataset(self.args.dataset)
        return trainloader, testloader, tensorboard_imgs


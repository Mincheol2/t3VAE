import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from torchvision.datasets import CelebA

class CustomCelebA(CelebA):
    def _check_integrity(self) -> bool:
        return True
    
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


class load_dataset():
    def __init__(self,batch_size,dataset,dataset_path):
        # data range : -1 ~ 1
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32))
        ]
        )
        self.batch_size = batch_size
        self.dataset = dataset
        self.path = dataset_path

    def load_celeb_dataset(self,dataset,dataset_path):

        self.transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(148),
            transforms.Resize(64),
            transforms.ToTensor(),
            ]
        )

        trainset = CustomCelebA(
            root=dataset_path,
            split='train',
            transform=self.transform,
            download=False,
        )
        
        testset = CustomCelebA(
            root=dataset_path,
            split='test',
            transform=self.transform,
            download=False,
        )
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=True)
        
        for images, _ in trainloader:
            sample_imgs = images
            break
        return trainloader, testloader, sample_imgs


    def select_dataloader(self):
        if 'celeb' in self.dataset:
            trainloader, testloader, tensorboard_imgs = self.load_celeb_dataset(self.dataset,self.path)
        elif 'cifar' in self.dataset:
            cifar_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.Resize(32),
            transforms.ToTensor(),
            ]
        )
            trainloader = datasets.CIFAR10(self.path, train=True, transform=cifar_transform, download = True)
            testloader = datasets.CIFAR10(self.path, train=False, transform=cifar_transform, download = True)
            tensorboard_imgs = None
            for images, _ in trainloader:
                tensorboard_imgs = images
                break
        else:
            raise Exception("Use appropriate dataset name")
        return trainloader, testloader, tensorboard_imgs
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

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
    def __init__(self,batch_size,dataset):
        # data range : -1 ~ 1
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((32, 32))
        ]
        )
        self.batch_size = batch_size
        self.dataset = dataset

    def load_celeb_dataset(self,dataset_name):

        self.transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(144),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        )
        train_img_list = []
        test_img_list = []
        celeb_total_nb = 202530

        # We manually split the train/test set.

        indices = np.random.permutation(celeb_total_nb) + 1
        train_indices = indices[:60000]
        test_indices = indices[60000:70001]

        if dataset_name == "celeb_crop64":
            for idx in train_indices:
                img_path = f"/data_intern/celeba/celeba_crop64/{idx:06d}.jpg"
                train_img_list.append(img_path)
            for idx in test_indices:
                img_path = f"/data_intern/celeba/celeba_crop64/{idx:06d}.jpg"
                test_img_list.append(img_path)
        elif dataset_name == "celeb_crop128":
            for idx in train_indices:
                img_path = f"/data_intern/celeba/celeba_crop128/{idx:06d}.jpg"
                train_img_list.append(img_path)
            for idx in test_indices:
                img_path = f"/data_intern/celeba/celeba_crop128/{idx:06d}.jpg"
                test_img_list.append(img_path)
        elif dataset_name == "celebA":
            for idx in train_indices:
                img_path = f"/data_intern/celeba/img_align_celeba/{idx:06d}.jpg"
                train_img_list.append(img_path)
            for idx in test_indices:
                img_path = f"/data_intern/celeba/img_align_celeba/{idx:06d}.jpg"
                test_img_list.append(img_path)
        else:
            raise Exception("Use proper size.")


        trainset = Custom_Dataset(file_list=train_img_list,
                            transform=self.transform)

        testset = Custom_Dataset(file_list=test_img_list,
                            transform=self.transform)
        ## Original data ##
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=True)
        
        for images, _ in trainloader:
            sample_imgs = images
            break
        return trainloader, testloader, sample_imgs


    def select_dataloader(self):
        if 'celeb' in self.dataset:
            trainloader, testloader, tensorboard_imgs = self.load_celeb_dataset(self.dataset)
        else:
            raise Exception("Use appropriate dataset name")
        return trainloader, testloader, tensorboard_imgs


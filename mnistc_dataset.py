import torch
import numpy as np
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms

class MYTensorDataset(torch.utils.data.Dataset):
    def __init__(self, *tensors) -> None:
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

class MNISTC_Dataset():
    def __init__(self, path='.', train=True, transform=None):
        self.path = path + '/mnist_c/'

          
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

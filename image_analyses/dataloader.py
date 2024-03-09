import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from torchvision.datasets import CelebA

class IMBALANCECIFAR10(datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='lt', imb_factor=100, train=True,
                 transform=None, target_transform=None):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, 1/imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'lt':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        
        new_data = np.vstack(new_data)
        print("the number of data:", len(new_data))
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


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
    def __init__(self,batch_size,dataset,dataset_path,imb_factor = None):
        # data range : -1 ~ 1
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32))
        ]
        )
        self.batch_size = batch_size
        self.dataset = dataset
        # self.path = dataset_path
        self.path = '/data_intern'
        self.imb_factor = imb_factor

    def load_celeb_dataset(self,dataset_path):

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

    def load_imb_cifar100_dataset(self,dataset_path, imb_factor=100):
        cifar_train_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                transforms.Resize(32),
                transforms.ToTensor(),
                ]
                )
        cifar_test_transform = transforms.Compose(
                [
                transforms.Resize(32),
                transforms.ToTensor(),
                ]
                )
        trainset = IMBALANCECIFAR100(root=dataset_path, train=True,
                imb_factor=imb_factor,transform=cifar_train_transform)
        testset = IMBALANCECIFAR100(root=dataset_path, train=False,
                imb_factor=1,transform=cifar_test_transform) # balanced testset
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=True)
        
        for images, _ in trainloader:
            sample_imgs = images
            break
        return trainloader, testloader, sample_imgs
    

    def load_imb_cifar10_dataset(self,dataset_path, imb_factor=100):
        cifar_train_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                transforms.Resize(32),
                transforms.ToTensor(),
                ]
                )
        cifar_test_transform = transforms.Compose(
                [
                transforms.Resize(32),
                transforms.ToTensor(),
                ]
                )
        trainset = IMBALANCECIFAR10(root=dataset_path, train=True,
                imb_factor=imb_factor,transform=cifar_train_transform)
        testset = datasets.CIFAR10(dataset_path, train=False, transform=cifar_test_transform, download = True) # balanced testset
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=True)
        
        for images, _ in trainloader:
            sample_imgs = images
            break
        return trainloader, testloader, sample_imgs


    def select_dataloader(self):
        if 'celeb' in self.dataset:
            trainloader, testloader, tensorboard_imgs = self.load_celeb_dataset(self.path)
        elif 'cifar' in self.dataset:
            if 'imb' in self.dataset:
                if '100' in self.dataset:
                    trainloader, testloader, tensorboard_imgs = self.load_imb_cifar100_dataset(self.path,imb_factor=self.imb_factor)
        
                elif '10' in self.dataset:
                    trainloader, testloader, tensorboard_imgs = self.load_imb_cifar10_dataset(self.path,imb_factor=self.imb_factor)
        
            else:
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

            mnist_transform = transforms.Compose(
            [            
            transforms.Resize(32),
            transforms.ToTensor(),
            ]
            )
            ## Imbalanced MNIST ##
            train_mnist = datasets.MNIST('/data_intern', train=True, transform=mnist_transform, download=True)
            noise_data = np.random.random((64,28,28))
            noise_label = np.ones(64) * 10

            train_mnist.data = np.concatenate([train_mnist.data, noise_data])
            train_mnist.targets = np.concatenate([train_mnist.targets, noise_label])

            trainloader = torch.utils.data.DataLoader(dataset=train_mnist,
                                            batch_size=self.batch_size,
                                            shuffle=True)



            test_mnist = datasets.MNIST('/data_intern', train=False, transform=mnist_transform, download=True)

            testloader = torch.utils.data.DataLoader(dataset=test_mnist,
                                            batch_size=self.batch_size,
                                            shuffle=True)

            tensorboard_imgs = None
            for images, _ in trainloader:
                tensorboard_imgs = images
                break
        else:
            raise Exception("Use appropriate dataset name")
        return trainloader, testloader, tensorboard_imgs
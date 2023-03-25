import torch
import torchvision
import random
import dataloader
import matplotlib.pyplot as plt
import argument
import numpy as np
import cv2
from sklearn.manifold import TSNE
from tqdm import tqdm

'''
   Deterministic operations are often slower than nondeterministic operations.
'''

args = argument.args
def make_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    


def make_tsne_plot(model, model_dir, DEVICE):
    '''
        draw a tsne plot and save it.
    '''
    args = argument.args

    if args.dataset in ['mnist', 'fashion']:
        test_class = model.testloader.dataset.targets.tolist()
    elif args.dataset == 'cifar10':
        test_class = model.testloader.dataset.targets
    else:
        # custom datasets
        test_class = model.testloader.dataset.tensors[1].tolist()
    # convert emnist to alphabets (e.g. -1-> a,  -7 -> g, -23 -> w)
    
    # convert emnist to alphabets (e.g. -1-> a,  -7 -> g, -23 -> w)
    
    if args.dataset == 'emnist':
        for i in range(len(test_class)):
            if test_class[i] < 0:
                test_class[i] = chr(-test_class[i] + 96)

    model.encoder.eval()
    model.decoder.eval()
    test_z = np.empty((0,args.zdim))
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(model.testloader):
            data = data.to(DEVICE)
            z, _, _ = model.encoder(data)
            z = z.detach().cpu().numpy()
            test_z = np.append(test_z,z,axis=0)


    colors_map = {}
    unique_labels = list(set(test_class))
    for i in range(len(unique_labels)):
        # outlier check
        if unique_labels[i] not in range(10):
            colors_map[unique_labels[i]] = -1
        else:
            colors_map[unique_labels[i]] = i
    
    colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
            "#A83683", "#4E655E", "#853541", "#3A3120","#535D8E","#c7bc22"]
    # outlier colors

    plt.rc('axes', unicode_minus=False)
    tsne = TSNE(random_state = args.seed)

    sample_size = 10000
    print(test_z.shape)

    tsne_z = tsne.fit_transform(test_z[0:sample_size])
    plt.figure(figsize=(10,10))
    plt.xlim(tsne_z[:,0].min(), tsne_z[:,0].max()+1)
    plt.ylim(tsne_z[:,1].min(), tsne_z[:,1].max()+1)

    for i in tqdm(range(sample_size)):
        class_name = str(abs(test_class[i]))
        plt.text(tsne_z[i,0], tsne_z[i,1], class_name,
                color = colors[colors_map[test_class[i]]],
                fontdict = {'weight':'bold','size':7})
    plt.xlabel("t-SNE 1st latent variable")
    plt.ylabel("t-SNE 2nd latent variable")

    if args.nu != 0 and args.beta == 0:
        plt.title(f"t-SNE : {args.dataset}, gammaAE nu = {args.nu}")
        filename = f'gammaAE_{args.dataset}_nu:{args.nu}_seed:{args.seed}'
    elif args.nu == 0 and args.beta != 0:
        plt.title(f"t-SNE : {args.dataset}, RVAE beta = {args.beta}")
        filename = f'RVAE_{args.dataset}_beta:{args.beta}_seed:{args.seed}'
    elif args.nu == 0 and args.beta == 0:
        plt.title(f"t-SNE : {args.dataset}, VAE")
        filename = f'VAE_{args.dataset}_seed:{args.seed}'

    plt.savefig(model_dir + f'/{filename}.png')



def make_square_masking(imgs, mask_ratio):
    '''
        input : imgs [B, C, H, W]
    '''
    B, _, H, W = imgs.shape
    blocks = []
    nb_blocks_H = 14
    nb_blocks_W = 14
    for i in range(nb_blocks_H):
        for j in range(nb_blocks_W):
            block_H = (H * i//nb_blocks_H, H * (i+1) // nb_blocks_H)
            block_W = (W * j//nb_blocks_W, W * (j+1) // nb_blocks_W)
            blocks.append((block_H,block_W))

    len_blocks = len(blocks)
    for k in range(B):
        rand_indices = np.random.choice(len_blocks, int(len_blocks * mask_ratio))
        mask_blocks = np.array(blocks)[rand_indices]
        for block in mask_blocks:
            imgs[k][block[0][0]:block[0][1], block[1][0]:block[1][1]] = 0

    return imgs


def sp_noise(image, prob):
    '''
        Add salt- pepper noise to image
        prob: Probability of the noise
    '''
    image = np.copy(image)
    black = 0
    white = 255
    rd_p = np.random.random(image.shape[:2])
    image[rd_p < (prob / 2)] = black
    image[rd_p > 1 - (prob / 2)] = white
    return image

def measure_sharpness(imgs):
    N = imgs.shape[0]
    sharpness = 0 
    for img in imgs:
        # 1. convert img to greyscale
        grey_img = torchvision.transforms.functional.rgb_to_grayscale(img).numpy()

        # 2. convolved with the laplace filter
        mask = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        laplacian = cv2.filter2D(grey_img, -1, mask)
        # 3.compute var of filtered img.
        sharpness += np.var(laplacian)

    return sharpness / N 

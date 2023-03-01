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
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
'''
    draw a tsne plot and save it.
'''

def make_tsne_plot(model, DEVICE):
    
    if args.dataset == "fashion":
        test_class = dataloader.testloader.dataset.targets.numpy()
        test_data = dataloader.testloader.dataset.data.unsqueeze(dim=1)
        test_z = model.encoder(test_data.to(dtype=torch.float32, device=DEVICE))[0]

    elif args.dataset == "mnist" :
        test_class = dataloader.testset.tensors[1]
        test_z = model.encoder(dataloader.testset.tensors[0].to(DEVICE))[0]
     
    else:
        test_class = dataloader.testset.datasets[0].dataset.tensors[1]
        test_z = model.encoder(dataloader.testset.datasets[0].dataset.tensors[0])[0]

    test_z = test_z.detach().cpu().numpy()

    colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
            "#A83683", "#4E655E", "#853541", "#3A3120","#535D8E"]

    plt.rc('axes', unicode_minus=False)
    tsne = TSNE(random_state = args.seed)

    sample_size = 10000

    tsne_z = tsne.fit_transform(test_z[0:sample_size])
    plt.figure(figsize=(10,10))
    plt.xlim(tsne_z[:,0].min(), tsne_z[:,0].max()+1)
    plt.ylim(tsne_z[:,1].min(), tsne_z[:,1].max()+1)
    for i in tqdm(range(sample_size)):
        plt.text(tsne_z[i,0], tsne_z[i,1], str(test_class[i]),
                color = colors[test_class[i]],
                fontdict = {'weight':'bold','size':7})
    plt.xlabel("t-SNE 1st latent variable")
    plt.ylabel("t-SNE 2nd latent variable")
    plt.title(f"t-SNE : {args.dataset}, nu = {args.nu}")

    if args.nu != 0 and args.beta == 0:
        filename = f'gammaAE_{args.dataset}_nu:{args.nu}_seed:{args.seed}'
    elif args.nu == 0 and args.beta != 0:
        filename = f'RVAE_{args.dataset}_beta:{args.beta}_seed:{args.seed}'
    elif args.nu == 0 and args.beta == 0:
        filename = f'VAE_{args.dataset}_nu:{args.nu}_seed:{args.seed}'


    plt.savefig(f'{filename}.png')



'''
    input : imgs [B, C, H, W]
'''
def make_masking(imgs, mask_ratio):
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


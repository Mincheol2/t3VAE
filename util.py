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
from torchmetrics import StructuralSimilarityIndexMeasure

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

    if args.dataset == "mnist" :
        test_class = dataloader.testset.tensors[1]
        test_z = model.encoder(dataloader.testset.tensors[0].to(DEVICE))[0]
     
    else:
        test_class = dataloader.testset.datasets[0].dataset.tensors[1]
        test_z = model.encoder(dataloader.testset.datasets[0].dataset.tensors[0].to(DEVICE))[0]

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

    plt.savefig(f"{args.dataset}_frac{args.frac}_nu{args.nu}_seed{args.seed}.png")

'''
Show reconstructed images.
'''
def show_images(images):
    images = torchvision.utils.make_grid(images)
    show_image(images)


def show_image(img):
    plt.imshow(img.permute(1, 2, 0), cmap="gray")
    plt.show()

'''
Caculate SSIM score
'''
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


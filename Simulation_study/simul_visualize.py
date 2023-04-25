import torch
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from simul_loss import log_t_normalizing_const
from simul_util import t_density, density_contour

class visualize() : 
    def __init__(self, p_dim) :
        self.visualize = visualize_PCA
        if p_dim == 3 : 
            self.visualize = visualize_3D
        elif p_dim == 2 : 
            self.visualize = visualize_2D
        elif p_dim == 1 : 
            self.visualize = visualize_density


def visualize_PCA(train_data, test_data, gAE_gen, VAE_gen, gAE_recon, VAE_recon, size = [9,6]) :
    train_data = train_data.cpu().numpy()
    test_data = test_data.cpu().numpy()
    gAE_gen = gAE_gen.cpu().numpy()
    VAE_gen = VAE_gen.cpu().numpy()
    gAE_recon = gAE_recon.cpu().numpy()
    VAE_recon = VAE_recon.cpu().numpy()
    
    pca = PCA(n_components=3)
    pca.fit(test_data)

    # PCA
    train_data = pca.transform(train_data) 
    test_data = pca.transform(test_data) 
    gAE_gen = pca.transform(gAE_gen.cpu().numpy()) 
    VAE_gen = pca.transform(VAE_gen.cpu().numpy()) 
    gAE_recon = pca.transform(gAE_recon.cpu().numpy()) 
    VAE_recon = pca.transform(VAE_recon.cpu().numpy()) 

    # plot
    fig = plt.figure(figsize = (size[0], size[1]))

    ax = fig.add_subplot(2,3,1, projection='3d')
    ax.scatter(test_data[:,0], test_data[:,1], test_data[:,2])
    domain = ax.axis()
    zlim = ax.get_zlim()
    plt.title('Test data')

    ax = fig.add_subplot(2,3,4, projection='3d')
    ax.scatter(train_data[:,0], train_data[:,1], train_data[:,2])
    ax.axis(domain)
    ax.set_zlim(zlim)
    plt.title('Train data')

    ax = fig.add_subplot(2,3,2, projection='3d')
    ax.scatter(gAE_gen[:,0], gAE_gen[:,1], gAE_gen[:,2])
    ax.axis(domain)
    ax.set_zlim(zlim)
    plt.title('gAE generation')

    ax = fig.add_subplot(2,3,5, projection='3d')
    ax.scatter(VAE_gen[:,0], VAE_gen[:,1], VAE_gen[:,2])
    ax.axis(domain)
    ax.set_zlim(zlim)
    plt.title('VAE generation')

    ax = fig.add_subplot(2,3,3, projection='3d')
    ax.scatter(gAE_recon[:,0], gAE_recon[:,1], gAE_recon[:,2])
    ax.axis(domain)
    ax.set_zlim(zlim)
    plt.title('gAE reconstruction')

    ax = fig.add_subplot(2,3,6, projection='3d')
    ax.scatter(VAE_recon[:,0], VAE_recon[:,1], VAE_recon[:,2])
    ax.axis(domain)
    ax.set_zlim(zlim)
    plt.title('VAE reconstruction')

    return fig

def visualize_3D(train_data, test_data, gAE_gen, VAE_gen, gAE_recon, VAE_recon, size = [9,6]) :
    train_data = train_data.cpu().numpy()
    test_data = test_data.cpu().numpy()
    gAE_gen = gAE_gen.cpu().numpy()
    VAE_gen = VAE_gen.cpu().numpy()
    gAE_recon = gAE_recon.cpu().numpy()
    VAE_recon = VAE_recon.cpu().numpy()

    # plot
    fig = plt.figure(figsize = (size[0], size[1]))

    ax = fig.add_subplot(2,3,1, projection='3d')
    ax.scatter(test_data[:,0], test_data[:,1], test_data[:,2])
    domain = ax.axis()
    zlim = ax.get_zlim()
    plt.title('Test data')

    ax = fig.add_subplot(2,3,4, projection='3d')
    ax.scatter(train_data[:,0], train_data[:,1], train_data[:,2])
    ax.axis(domain)
    ax.set_zlim(zlim)
    plt.title('Train data')

    ax = fig.add_subplot(2,3,2, projection='3d')
    ax.scatter(gAE_gen[:,0], gAE_gen[:,1], gAE_gen[:,2])
    ax.axis(domain)
    ax.set_zlim(zlim)
    plt.title('gAE generation')

    ax = fig.add_subplot(2,3,5, projection='3d')
    ax.scatter(VAE_gen[:,0], VAE_gen[:,1], VAE_gen[:,2])
    ax.axis(domain)
    ax.set_zlim(zlim)
    plt.title('VAE generation')

    ax = fig.add_subplot(2,3,3, projection='3d')
    ax.scatter(gAE_recon[:,0], gAE_recon[:,1], gAE_recon[:,2])
    ax.axis(domain)
    ax.set_zlim(zlim)
    plt.title('gAE reconstruction')

    ax = fig.add_subplot(2,3,6, projection='3d')
    ax.scatter(VAE_recon[:,0], VAE_recon[:,1], VAE_recon[:,2])
    ax.axis(domain)
    ax.set_zlim(zlim)
    plt.title('VAE reconstruction')

    return fig


def visualize_2D(train_data, test_data, gAE_gen, VAE_gen, gAE_recon, VAE_recon, size = [9,6]) :
    train_data = train_data.cpu().numpy()
    test_data = test_data.cpu().numpy()
    gAE_gen = gAE_gen.cpu().numpy()
    VAE_gen = VAE_gen.cpu().numpy()
    gAE_recon = gAE_recon.cpu().numpy()
    VAE_recon = VAE_recon.cpu().numpy()

    # plot
    fig = plt.figure(figsize = (size[0], size[1]))

    ax = fig.add_subplot(2,3,1)
    ax.scatter(test_data[:,0], test_data[:,1])
    domain = ax.axis()
    plt.title('Test data')

    ax = fig.add_subplot(2,3,4)
    ax.scatter(train_data[:,0], train_data[:,1])
    ax.axis(domain)
    plt.title('Train data')

    ax = fig.add_subplot(2,3,2)
    ax.scatter(gAE_gen[:,0], gAE_gen[:,1])
    ax.axis(domain)
    plt.title('gAE generation')

    ax = fig.add_subplot(2,3,5)
    ax.scatter(VAE_gen[:,0], VAE_gen[:,1])
    ax.axis(domain)
    plt.title('VAE generation')

    ax = fig.add_subplot(2,3,3)
    ax.scatter(gAE_recon[:,0], gAE_recon[:,1])
    ax.axis(domain)
    plt.title('gAE reconstruction')

    ax = fig.add_subplot(2,3,6)
    ax.scatter(VAE_recon[:,0], VAE_recon[:,1])
    ax.axis(domain)
    plt.title('VAE reconstruction')

    return fig

def visualize_density(train_data, test_data, model_nu_list, gAE_gen_list, VAE_gen, K, sample_nu_list, mu_list, var_list, ratio_list) :
    train_data = train_data.cpu().squeeze(1).numpy()
    test_data = test_data.cpu().squeeze(1).numpy()
    gAE_gen_list = [gAE_gen.cpu().squeeze(1).numpy() for gAE_gen in gAE_gen_list]
    VAE_gen = VAE_gen.cpu().squeeze(1).numpy()

    M = len(gAE_gen_list)

    input = np.arange(-20000, 20001) * 0.01
    contour = density_contour(input, K, sample_nu_list, mu_list, var_list, ratio_list).squeeze().numpy()

    # plot
    fig = plt.figure(figsize = (4 * (M+2), 8))

    ax = fig.add_subplot(2,M+2,1)
    plt.plot(input, contour)
    plt.hist(train_data, bins = 100, range = [-20, 20], alpha = 0.5, density=True, label = "Train")
    plt.hist(test_data, bins = 100, range = [-20, 20], alpha = 0.5, density=True, label = "Test")
    plt.xlim(-20, 20)
    plt.title('Train and test data')

    ax = fig.add_subplot(2,M+2,M+3)
    plt.plot(input, contour)
    plt.hist(train_data, bins = 100, range = [-200, 200], alpha = 0.5, density=True, label = "Train")
    plt.hist(test_data, bins = 100, range = [-200, 200], alpha = 0.5, density=True, label = "Test")
    plt.xlim(-200, 200)
    plt.yscale("log")
    plt.title('Train and test data (log scale)')

    for m in range(M) : 
        ax = fig.add_subplot(2,M+2,m+2)
        plt.plot(input, contour)
        plt.hist(gAE_gen_list[m], bins = 100, range = [-20, 20], density=True)
        plt.xlim(-20, 20)
        plt.title(f'gAE generation (nu = {model_nu_list[m]})')

        ax = fig.add_subplot(2,M+2,M+m+4)
        plt.plot(input, contour)
        plt.hist(gAE_gen_list[m], bins = 100, range = [-200, 200], density=True)
        plt.xlim(-200, 200)
        plt.yscale("log")
        plt.title(f'gAE generation (nu = {model_nu_list[m]}, log scale)')

    ax = fig.add_subplot(2,M+2,M+2)
    plt.plot(input, contour)
    plt.hist(VAE_gen, bins = 100, range = [-20, 20], density=True)
    plt.xlim(-20, 20)
    plt.title('VAE generation')

    ax = fig.add_subplot(2,M+2,2*M+4)
    plt.plot(input, contour)
    plt.hist(VAE_gen, bins = 100, range = [-200, 200], density=True)
    plt.xlim(-200, 200)
    plt.yscale("log")
    plt.title('VAE generation (log scale)')

    return fig


    



# def visualize_density_old(train_data, test_data, gAE_gen, VAE_gen, gAE_recon, VAE_recon, size = [9,6]) :
#     train_data = train_data.cpu().numpy()
#     test_data = test_data.cpu().numpy()
#     gAE_gen = gAE_gen.cpu().numpy()
#     VAE_gen = VAE_gen.cpu().numpy()
#     gAE_recon = gAE_recon.cpu().numpy()
#     VAE_recon = VAE_recon.cpu().numpy()

#     # plot
#     fig = plt.figure(figsize = (size[0], size[1]))

#     ax = fig.add_subplot(2,3,1)
#     sns.kdeplot(train_data.squeeze(), color = "Green", label = "Train")
#     sns.kdeplot(test_data.squeeze(), color = "Blue", label = "Test")
#     domain = ax.axis()
#     sns.kdeplot(gAE_gen.squeeze(), color = "Red", label = "gAE_gen")
#     ax.axis(domain)
#     ax.set_xlim(-20,20)
#     plt.title('gAE generation')
#     # plt.legend()

#     ax = fig.add_subplot(2,3,4)
#     sns.kdeplot(train_data.squeeze(), color = "Green", label = "Train")
#     sns.kdeplot(test_data.squeeze(), color = "Blue", label = "Test")
#     sns.kdeplot(VAE_gen.squeeze(), color = "Red", label = "VAE_gen")
#     ax.axis(domain)
#     ax.set_xlim(-20,20)
#     plt.title('VAE generation')   

    
#     ax = fig.add_subplot(2,3,2)
#     sns.kdeplot(train_data.squeeze(), color = "Green", label = "Train")
#     sns.kdeplot(test_data.squeeze(), color = "Blue", label = "Test")
#     domain = ax.axis()
#     sns.kdeplot(gAE_gen.squeeze(), color = "Red", label = "gAE_gen")
#     ax.set_xlim(-20,20)
#     plt.yscale("log")
#     plt.ylim(1e-6, 1e1)
#     plt.title('gAE generation')
#     # plt.legend()

#     ax = fig.add_subplot(2,3,5)
#     sns.kdeplot(train_data.squeeze(), color = "Green", label = "Train")
#     sns.kdeplot(test_data.squeeze(), color = "Blue", label = "Test")
#     sns.kdeplot(VAE_gen.squeeze(), color = "Red", label = "VAE_gen")
#     ax.set_xlim(-20,20)
#     plt.yscale("log")
#     plt.ylim(1e-6, 1e1)
#     plt.title('VAE generation')   
    
#     ax = fig.add_subplot(2,3,3)
#     sns.kdeplot(train_data.squeeze(), color = "Green", label = "Train")
#     sns.kdeplot(test_data.squeeze(), color = "Blue", label = "Test")
#     sns.kdeplot(gAE_recon.squeeze(), color = "Red", label = "gAE_recon")
#     ax.axis(domain)
#     ax.set_xlim(-20,20)
#     plt.title('gAE reconstruction')    
#     # plt.legend()
    
#     ax = fig.add_subplot(2,3,6)
#     sns.kdeplot(train_data.squeeze(), color = "Green", label = "Train")
#     sns.kdeplot(test_data.squeeze(), color = "Blue", label = "Test")
#     sns.kdeplot(VAE_recon.squeeze(), color = "Red", label = "VAE_recon")
#     ax.axis(domain)
#     ax.set_xlim(-20,20)
#     plt.title('VAE reconstruction')
#     # plt.legend()

#     return fig


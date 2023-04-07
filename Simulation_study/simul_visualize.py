import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class visualize() : 
    def __init__(self, p_dim) :
        self.visualize = visualize_PCA
        if p_dim == 3 : 
            self.visualize = visualize_3D
        elif p_dim == 2 : 
            self.visualize = visualize_2D


def visualize_PCA(train_data, test_data, gAE_gen, VAE_gen, gAE_recon, VAE_recon, size = [9,6]) :
    train_data = train_data.cpu().numpy()
    test_data = test_data.cpu().numpy()
    
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
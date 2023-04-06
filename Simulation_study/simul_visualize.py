import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def visualize_PCA(data, size = [6,8], TITLE = 'Sample') : 
    data = data.cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit(data)
    data_pca = pca.transform(data) 
    
    fig = plt.figure(figsize=(size[0], size[1]))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(data_pca[:,0], data_pca[:,1], data_pca[:,2])

    plt.title(TITLE)

    return fig

def visualize_3D(data, axis = [0,1,2], size = [6,8], TITLE = 'Sample') : 
    data = data.cpu().numpy()
    fig = plt.figure(figsize=(size[0], size[1]))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(data[:,axis[0]], 
               data[:,axis[1]], 
               data[:,axis[2]])

    plt.title(TITLE)

    return fig

def visualize_2D(data, axis = [0,1], size = [6,8], TITLE = 'Sample') : 
    data = data.cpu().numpy()
    fig = plt.figure(figsize=(size[0], size[1]))
    ax = fig.add_subplot()

    ax.scatter(data[:,axis[0]], 
               data[:,axis[1]])

    plt.title(TITLE)

    return fig

def total_visualize_PCA(data, gAE_recon, gAE_gen, VAE_recon, VAE_gen, size = [12,8]) :
    data = data.cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit(data)
    data_pca = pca.transform(data) 
    gAE_recon_pca = pca.transform(gAE_recon) 
    gAE_gen_pca = pca.transform(gAE_gen) 
    VAE_recon_pca = pca.transform(VAE_recon) 
    VAE_gen_pca = pca.transform(VAE_gen) 

    fig = plt.figure(figsize = (size[0], size[1]))

    ax = fig.add_subplot(2,3,1, projection='3d')
    ax.scatter(data_pca[:,0], data_pca[:,1], data_pca[:,2])
    domain = ax.axis()
    plt.title('sample')

    ax = fig.add_subplot(2,3,2, projection='3d')
    ax.scatter(VAE_gen[:,0], VAE_gen[:,1], VAE_gen[:,2])
    ax.axis(domain)
    plt.title('gammaAE generation')

    ax = fig.add_subplot(2,3,3, projection='3d')
    ax.scatter(data_pca[:,0], data_pca[:,1], data_pca[:,2])
    ax.axis(domain)
    plt.title('VAE generation')

    ax = fig.add_subplot(2,3,5, projection='3d')
    ax.scatter(gAE_recon[:,0], gAE_recon[:,1], gAE_recon[:,2])
    ax.axis(domain)
    plt.title('gammaAE reconstruction')

    ax = fig.add_subplot(2,3,6, projection='3d')
    ax.scatter(VAE_recon[:,0], VAE_recon[:,1], VAE_recon[:,2])
    ax.axis(domain)
    plt.title('VAE reconstruction')

    return fig

def total_visualize_3D(data, gAE_recon, gAE_gen, VAE_recon, VAE_gen, axis = [0,1,2], size = [12,8]) :
    fig = plt.figure(figsize = (size[0], size[1]))
    ax = fig.add_subplot(2,3,1, projection='3d')
    ax.scatter(data.cpu().numpy()[:,axis[0]], 
               data.cpu().numpy()[:,axis[1]], 
               data.cpu().numpy()[:,axis[2]])
    domain = ax.axis()
    plt.title('sample')

    ax = fig.add_subplot(2,3,2, projection='3d')
    ax.scatter(gAE_gen[:,axis[0]], 
               gAE_gen[:,axis[1]], 
               gAE_gen[:,axis[2]])
    ax.axis(domain)
    plt.title('gammaAE generation')

    ax = fig.add_subplot(2,3,3, projection='3d')
    ax.scatter(VAE_gen[:,axis[0]], 
               VAE_gen[:,axis[1]], 
               VAE_gen[:,axis[2]])
    ax.axis(domain)
    plt.title('VAE generation')

    ax = fig.add_subplot(2,3,5, projection='3d')
    ax.scatter(gAE_recon[:,axis[0]], 
               gAE_recon[:,axis[1]], 
               gAE_recon[:,axis[2]])
    ax.axis(domain)
    plt.title('gammaAE reconstruction')

    ax = fig.add_subplot(2,3,6, projection='3d')
    ax.scatter(VAE_recon[:,axis[0]], 
               VAE_recon[:,axis[1]], 
               VAE_recon[:,axis[2]])
    ax.axis(domain)
    plt.title('VAE reconstruction')

    return fig
    
def total_visualize_2D(data, gAE_recon, gAE_gen, VAE_recon, VAE_gen, axis = [0,1], size = [12,8]) :
    fig = plt.figure(figsize = (size[0], size[1]))
    ax = fig.add_subplot(2,3,1)
    ax.scatter(data.cpu().numpy()[:,axis[0]], 
               data.cpu().numpy()[:,axis[1]])
    domain = ax.axis()
    plt.title('sample')

    ax = fig.add_subplot(2,3,2)
    ax.scatter(gAE_gen[:,axis[0]], 
               gAE_gen[:,axis[1]])
    ax.axis(domain)
    plt.title('gammaAE generation')

    ax = fig.add_subplot(2,3,3)
    ax.scatter(VAE_gen[:,axis[0]], 
               VAE_gen[:,axis[1]])
    ax.axis(domain)
    plt.title('VAE generation')

    ax = fig.add_subplot(2,3,5)
    ax.scatter(gAE_recon[:,axis[0]], 
               gAE_recon[:,axis[1]])
    ax.axis(domain)
    plt.title('gammaAE reconstruction')

    ax = fig.add_subplot(2,3,6)
    ax.scatter(VAE_recon[:,axis[0]], 
               VAE_recon[:,axis[1]])
    ax.axis(domain)
    plt.title('VAE reconstruction')

    return fig
    


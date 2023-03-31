import matplotlib.pyplot as plt
import numpy as np

def visualize_3D(data, axis = [0,1,2], size = [6,8], TITLE = 'Sample') : 
    fig = plt.figure(figsize=(size[0], size[1]))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(data.cpu().numpy()[:,axis[0]], 
               data.cpu().numpy()[:,axis[1]], 
               data.cpu().numpy()[:,axis[2]])
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')

    plt.title(TITLE)

    return fig


def visualize_2D(data, axis = [0,1], size = [6,8], TITLE = 'Sample') : 
    fig = plt.figure(figsize=(size[0], size[1]))
    ax = fig.add_subplot()

    ax.scatter(data.cpu().numpy()[:,axis[0]], 
               data.cpu().numpy()[:,axis[1]])
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')

    plt.title(TITLE)

    return fig



def total_visualize_3D(data, gAE_recon, gAE_gen, VAE_recon, VAE_gen, axis = [0,1,2], size = [12,8]) :
    fig = plt.figure(figsize = (size[0], size[1]))
    ax = fig.add_subplot(2,3,1, projection='3d')
    ax.scatter(data.cpu().numpy()[:,axis[0]], 
               data.cpu().numpy()[:,axis[1]], 
               data.cpu().numpy()[:,axis[2]])
    plt.title('sample')

    ax = fig.add_subplot(2,3,2, projection='3d')
    ax.scatter(gAE_gen[:,axis[0]], 
               gAE_gen[:,axis[1]], 
               gAE_gen[:,axis[2]])
    plt.title('gammaAE generation')

    ax = fig.add_subplot(2,3,3, projection='3d')
    ax.scatter(VAE_gen[:,axis[0]], 
               VAE_gen[:,axis[1]], 
               VAE_gen[:,axis[2]])
    plt.title('VAE generation')

    ax = fig.add_subplot(2,3,5, projection='3d')
    ax.scatter(gAE_recon[:,axis[0]], 
               gAE_recon[:,axis[1]], 
               gAE_recon[:,axis[2]])
    plt.title('gammaAE reconstruction')

    ax = fig.add_subplot(2,3,6, projection='3d')
    ax.scatter(VAE_recon[:,axis[0]], 
               VAE_recon[:,axis[1]], 
               VAE_recon[:,axis[2]])
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
    


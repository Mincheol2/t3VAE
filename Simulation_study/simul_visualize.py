import torch
import numpy as np
import matplotlib.pyplot as plt

# import seaborn as sns
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

from simul_loss import log_t_normalizing_const
from simul_util import t_density, t_density_contour

def visualize_density(train_data, test_data, model_nu_list, 
                      gAE_gen_list, VAE_gen, gAE_recon_list, VAE_recon, 
                      K, sample_nu_list, mu_list, var_list, ratio_list, xlim = 200) :
    train_data = train_data.cpu().squeeze(1).numpy()
    test_data = test_data.cpu().squeeze(1).numpy()
    gAE_gen_list = [gAE_gen[torch.isfinite(gAE_gen)].cpu().numpy() for gAE_gen in gAE_gen_list]
    VAE_gen = VAE_gen[torch.isfinite(VAE_gen)].cpu().numpy()
    gAE_recon_list = [gAE_recon[torch.isfinite(gAE_recon)].cpu().numpy() for gAE_recon in gAE_recon_list]
    VAE_recon = VAE_recon[torch.isfinite(VAE_recon)].cpu().numpy()

    M = len(gAE_gen_list)
    input = np.arange(-xlim * 100, xlim * 100 + 1) * 0.01
    contour = t_density_contour(input, K, sample_nu_list, mu_list, var_list, ratio_list).squeeze().numpy()

    # plot
    fig = plt.figure(figsize = (3 * (M+2), 12))

    ax = fig.add_subplot(4,M+2,1)
    plt.plot(input, contour)
    plt.hist(train_data, bins = 100, range = [-20, 20], alpha = 0.5, density=True, label = "Train")
    plt.xlim(-20, 20)
    plt.title(f'Train data (nu = {sample_nu_list})')

    ax = fig.add_subplot(4,M+2,M+3)
    plt.plot(input, contour)
    plt.hist(train_data, bins = 100, range = [-xlim, xlim], alpha = 0.5, density=True, label = "Train")
    plt.xlim(-xlim, xlim)
    plt.yscale("log")
    plt.title('Train data (log scale)')

    ax = fig.add_subplot(4,M+2,2*M+5)   
    plt.plot(input, contour)
    plt.hist(test_data, bins = 100, range = [-20, 20], alpha = 0.5, density=True, label = "Test")
    plt.xlim(-20, 20)
    plt.title(f'Test data (nu = {sample_nu_list})')

    ax = fig.add_subplot(4,M+2,3*M+7)
    plt.plot(input, contour)
    plt.hist(test_data, bins = 100, range = [-xlim, xlim], alpha = 0.5, density=True, label = "Test")
    plt.xlim(-xlim, xlim)
    plt.yscale("log")
    plt.title('Test data (log scale)')

    for m in range(M) : 
        ax = fig.add_subplot(4,M+2,m+2)
        plt.plot(input, contour)
        plt.hist(gAE_gen_list[m], bins = 100, range = [-20, 20], density=True)
        plt.xlim(-20, 20)
        plt.title(f'gAE generation (nu = {model_nu_list[m]})')

        ax = fig.add_subplot(4,M+2,M+m+4)
        plt.plot(input, contour)
        plt.hist(gAE_gen_list[m], bins = 100, range = [-xlim, xlim], density=True)
        plt.xlim(-xlim, xlim)
        plt.yscale("log")
        plt.title(f'gAE generation (log scale)')

        ax = fig.add_subplot(4,M+2,2*M+m+6)
        plt.plot(input, contour)
        plt.hist(gAE_recon_list[m], bins = 100, range = [-20, 20], density=True)
        plt.xlim(-20, 20)
        plt.title(f'gAE reconstruction')

        ax = fig.add_subplot(4,M+2,3*M+m+8)
        plt.plot(input, contour)
        plt.hist(gAE_recon_list[m], bins = 100, range = [-xlim, xlim], density=True)
        plt.xlim(-xlim, xlim)
        plt.yscale("log")
        plt.title(f'gAE reconstruction (log scale)')

    ax = fig.add_subplot(4,M+2,M+2)
    plt.plot(input, contour)
    plt.hist(VAE_gen, bins = 100, range = [-20, 20], density=True)
    plt.xlim(-20, 20)
    plt.title('VAE generation')

    ax = fig.add_subplot(4,M+2,2*M+4)
    plt.plot(input, contour)
    plt.hist(VAE_gen, bins = 100, range = [-xlim, xlim], density=True)
    plt.xlim(-xlim, xlim)
    plt.yscale("log")
    plt.title('VAE generation (log scale)')

    ax = fig.add_subplot(4,M+2,3*M+6)
    plt.plot(input, contour)
    plt.hist(VAE_recon, bins = 100, range = [-20, 20], density=True)
    plt.xlim(-20, 20)
    plt.title('VAE reconstruction')

    ax = fig.add_subplot(4,M+2,4*M+8)
    plt.plot(input, contour)
    plt.hist(VAE_recon, bins = 100, range = [-xlim, xlim], density=True)
    plt.xlim(-xlim, xlim)
    plt.yscale("log")
    plt.title('VAE reconstruction (log scale)')

    return fig

def visualize_density_simple(model_nu_list, gAE_gen_list, VAE_gen, 
                             K, sample_nu_list, mu_list, var_list, ratio_list, xlim = 200) :
    gAE_gen_list = [gAE_gen[torch.isfinite(gAE_gen)].cpu().numpy() for gAE_gen in gAE_gen_list]
    VAE_gen = VAE_gen[torch.isfinite(VAE_gen)].cpu().numpy()

    M = len(gAE_gen_list)
    input = np.arange(-xlim * 100, xlim * 100 + 1) * 0.01
    contour = t_density_contour(input, K, sample_nu_list, mu_list, var_list, ratio_list).squeeze().numpy()

    # plot
    fig = plt.figure(figsize = (3.5 * (M+1), 7))

    ax = fig.add_subplot(2,M+1,1)
    plt.plot(input, contour, color='black')
    plt.hist(VAE_gen, bins = 100, range = [-10, 10], density=True, alpha = 0.5, color='dodgerblue')
    plt.xlim(-10, 10)
    plt.title('VAE')

    ax = fig.add_subplot(2,M+1,M+2)
    plt.plot(input, contour, color='black')
    plt.hist(VAE_gen, bins = 100, range = [-xlim, xlim], density=True, alpha = 0.5, color='dodgerblue')
    plt.xlim(-xlim, xlim)
    plt.yscale("log")
    plt.ylim(1e-6, 1)

    for m in range(M) : 
        ax = fig.add_subplot(2,M+1,m+2)
        plt.plot(input, contour, color='black')
        plt.hist(gAE_gen_list[m], bins = 100, range = [-10, 10], density=True, alpha = 0.5, color='dodgerblue')
        plt.xlim(-10, 10)
        plt.title(f't3VAE (nu = {model_nu_list[m]})')

        ax = fig.add_subplot(2,M+1,M+m+3)
        plt.plot(input, contour, color='black')
        plt.hist(gAE_gen_list[m], bins = 100, range = [-xlim, xlim], density=True, alpha = 0.5, color='dodgerblue')
        plt.xlim(-xlim, xlim)
        plt.yscale("log")
        plt.ylim(1e-6, 1)

    # plt.savefig("heavy_tail.png")
    return fig


# def visualize_latent(sample_nu, test_latent, test_data, model_nu_list, gAE_latent_list, VAE_latent, gAE_recon_list, VAE_recon, xlim) :
#     test_latent = test_latent.cpu().numpy()
#     test_data = test_data.cpu().numpy()
#     gAE_latent_list = [gAE_latent.cpu().numpy() for gAE_latent in gAE_latent_list]
#     VAE_latent = VAE_latent.cpu().numpy()
#     gAE_recon_list = [gAE_recon.cpu().numpy() for gAE_recon in gAE_recon_list]
#     VAE_recon = VAE_recon.cpu().numpy()
    
#     M = len(gAE_latent_list)

#     # plot
#     fig = plt.figure(figsize = (3 * (M+2), 6))

#     ax = fig.add_subplot(2,M+2,1)
#     ax.scatter(test_latent[:,0], test_latent[:,1])
#     domain1 = ax.axis()
#     plt.title(f'True latent (nu = {sample_nu})')

#     ax = fig.add_subplot(2,M+2,M+3, projection='3d')
#     ax.scatter(test_data[:,0], test_data[:,1], test_data[:,2])
#     domain2 = ax.axis()
#     zlim = ax.get_zlim()
#     plt.title(f'True data')

#     for m in range(M) : 
#         ax = fig.add_subplot(2,M+2,m+2)
#         ax.scatter(gAE_latent_list[m][:,0], gAE_latent_list[m][:,1])
#         ax.axis(domain1)
#         plt.title(f'gAE latent (nu = {model_nu_list[m]})')

#         ax = fig.add_subplot(2,M+2,M+m+4, projection='3d')
#         ax.scatter(gAE_recon_list[m][:,0], gAE_recon_list[m][:,1], gAE_recon_list[m][:,2])
#         ax.axis(domain2)
#         ax.set_zlim(zlim)
#         plt.title(f'gAE reconstruction')

#     ax = fig.add_subplot(2,M+2,M+2)
#     ax.scatter(VAE_latent[:,0], VAE_latent[:,1])
#     ax.axis(domain1)
#     plt.title(f'VAE latent (nu = {model_nu_list[m]})')

#     ax = fig.add_subplot(2,M+2,2*M+4, projection='3d')
#     ax.scatter(VAE_recon[:,0], VAE_recon[:,1], VAE_recon[:,2])
#     ax.axis(domain2)
#     ax.set_zlim(zlim)
#     plt.title(f'VAE reconstruction')

#     return fig

# def visualize_latent_2D(sample_nu, test_latent, test_data, model_nu_list, gAE_latent_list, VAE_latent, gAE_recon_list, VAE_recon, xlim = 20) :
#     test_latent = test_latent.cpu().numpy()
#     test_data = test_data.cpu().numpy()
#     gAE_latent_list = [gAE_latent.cpu().numpy() for gAE_latent in gAE_latent_list]
#     VAE_latent = VAE_latent.cpu().numpy()
#     gAE_recon_list = [gAE_recon.cpu().numpy() for gAE_recon in gAE_recon_list]
#     VAE_recon = VAE_recon.cpu().numpy()
    
#     M = len(gAE_latent_list)

#     # plot
#     fig = plt.figure(figsize = (3 * (M+2), 6))

#     ax = fig.add_subplot(2,M+2,1)
#     ax.hist(test_latent, bins = 100, range = [-xlim, xlim], density = True, label = 'Test latent')
#     ax.xlim(-xlim, xlim)
#     plt.title(f'True latent (nu = {sample_nu})')

#     ax = fig.add_subplot(2,M+2,M+3)
#     ax.scatter(test_data[:,0], test_data[:,1])
#     domain = ax.axis()
#     ylim = ax.get_zlim()
#     plt.title(f'True data')

#     for m in range(M) : 
#         ax = fig.add_subplot(2,M+2,m+2)
#         ax.hist(gAE_latent_list[m][:,0], )
#         ax.scatter(gAE_latent_list[m][:,0], bins = 100, range = [-xlim, xlim], density = True)
#         ax.xlim(-xlim, xlim)
#         plt.title(f'gAE latent (nu = {model_nu_list[m]})')

#         ax = fig.add_subplot(2,M+2,M+m+4)
#         ax.scatter(gAE_recon_list[m][:,0], gAE_recon_list[m][:,1])
#         ax.axis(domain)
#         ax.set_zlim(ylim)
#         plt.title(f'gAE reconstruction')

#     ax = fig.add_subplot(2,M+2,M+2)
#     ax.hist(VAE_latent[:,0], bins = 100, range = [-xlim, xlim], density = True)
#     ax.xlim(-xlim, xlim)
#     plt.title(f'VAE latent (nu = {model_nu_list[m]})')

#     ax = fig.add_subplot(2,M+2,2*M+4)
#     ax.scatter(VAE_recon[:,0], VAE_recon[:,1])
#     ax.axis(domain)
#     ax.set_zlim(ylim)
#     plt.title(f'VAE reconstruction')

#     return fig



# def visualize_3D(train_data, test_data, gAE_gen, VAE_gen, gAE_recon, VAE_recon, size = [9,6]) :
#     train_data = train_data.cpu().numpy()
#     test_data = test_data.cpu().numpy()
#     gAE_gen = gAE_gen.cpu().numpy()
#     VAE_gen = VAE_gen.cpu().numpy()
#     gAE_recon = gAE_recon.cpu().numpy()
#     VAE_recon = VAE_recon.cpu().numpy()

#     # plot
#     fig = plt.figure(figsize = (size[0], size[1]))

#     ax = fig.add_subplot(2,3,1, projection='3d')
#     ax.scatter(test_data[:,0], test_data[:,1], test_data[:,2])
#     domain = ax.axis()
#     zlim = ax.get_zlim()
#     plt.title('Test data')

#     ax = fig.add_subplot(2,3,4, projection='3d')
#     ax.scatter(train_data[:,0], train_data[:,1], train_data[:,2])
#     ax.axis(domain)
#     ax.set_zlim(zlim)
#     plt.title('Train data')

#     ax = fig.add_subplot(2,3,2, projection='3d')
#     ax.scatter(gAE_gen[:,0], gAE_gen[:,1], gAE_gen[:,2])
#     ax.axis(domain)
#     ax.set_zlim(zlim)
#     plt.title('gAE generation')

#     ax = fig.add_subplot(2,3,5, projection='3d')
#     ax.scatter(VAE_gen[:,0], VAE_gen[:,1], VAE_gen[:,2])
#     ax.axis(domain)
#     ax.set_zlim(zlim)
#     plt.title('VAE generation')

#     ax = fig.add_subplot(2,3,3, projection='3d')
#     ax.scatter(gAE_recon[:,0], gAE_recon[:,1], gAE_recon[:,2])
#     ax.axis(domain)
#     ax.set_zlim(zlim)
#     plt.title('gAE reconstruction')

#     ax = fig.add_subplot(2,3,6, projection='3d')
#     ax.scatter(VAE_recon[:,0], VAE_recon[:,1], VAE_recon[:,2])
#     ax.axis(domain)
#     ax.set_zlim(zlim)
#     plt.title('VAE reconstruction')

#     return fig




# def visualize_2D(train_data, test_data, gAE_gen, VAE_gen, gAE_recon, VAE_recon, size = [9,6]) :
#     train_data = train_data.cpu().numpy()
#     test_data = test_data.cpu().numpy()
#     gAE_gen = gAE_gen.cpu().numpy()
#     VAE_gen = VAE_gen.cpu().numpy()
#     gAE_recon = gAE_recon.cpu().numpy()
#     VAE_recon = VAE_recon.cpu().numpy()

#     # plot
#     fig = plt.figure(figsize = (size[0], size[1]))

#     ax = fig.add_subplot(2,3,1)
#     ax.scatter(test_data[:,0], test_data[:,1])
#     domain = ax.axis()
#     plt.title('Test data')

#     ax = fig.add_subplot(2,3,4)
#     ax.scatter(train_data[:,0], train_data[:,1])
#     ax.axis(domain)
#     plt.title('Train data')

#     ax = fig.add_subplot(2,3,2)
#     ax.scatter(gAE_gen[:,0], gAE_gen[:,1])
#     ax.axis(domain)
#     plt.title('gAE generation')

#     ax = fig.add_subplot(2,3,5)
#     ax.scatter(VAE_gen[:,0], VAE_gen[:,1])
#     ax.axis(domain)
#     plt.title('VAE generation')

#     ax = fig.add_subplot(2,3,3)
#     ax.scatter(gAE_recon[:,0], gAE_recon[:,1])
#     ax.axis(domain)
#     plt.title('gAE reconstruction')

#     ax = fig.add_subplot(2,3,6)
#     ax.scatter(VAE_recon[:,0], VAE_recon[:,1])
#     ax.axis(domain)
#     plt.title('VAE reconstruction')

#     return fig



# class visualize() : 
#     def __init__(self, p_dim) :
#         self.visualize = visualize_PCA
#         if p_dim == 3 : 
#             self.visualize = visualize_3D
#         elif p_dim == 2 : 
#             self.visualize = visualize_2D
#         elif p_dim == 1 : 
#             self.visualize = visualize_density




# def visualize_PCA(train_data, test_data, gAE_gen, VAE_gen, gAE_recon, VAE_recon, size = [9,6]) :
#     train_data = train_data.cpu().numpy()
#     test_data = test_data.cpu().numpy()
#     gAE_gen = gAE_gen.cpu().numpy()
#     VAE_gen = VAE_gen.cpu().numpy()
#     gAE_recon = gAE_recon.cpu().numpy()
#     VAE_recon = VAE_recon.cpu().numpy()
    
#     pca = PCA(n_components=3)
#     pca.fit(test_data)

#     # PCA
#     train_data = pca.transform(train_data) 
#     test_data = pca.transform(test_data) 
#     gAE_gen = pca.transform(gAE_gen.cpu().numpy()) 
#     VAE_gen = pca.transform(VAE_gen.cpu().numpy()) 
#     gAE_recon = pca.transform(gAE_recon.cpu().numpy()) 
#     VAE_recon = pca.transform(VAE_recon.cpu().numpy()) 

#     # plot
#     fig = plt.figure(figsize = (size[0], size[1]))

#     ax = fig.add_subplot(2,3,1, projection='3d')
#     ax.scatter(test_data[:,0], test_data[:,1], test_data[:,2])
#     domain = ax.axis()
#     zlim = ax.get_zlim()
#     plt.title('Test data')

#     ax = fig.add_subplot(2,3,4, projection='3d')
#     ax.scatter(train_data[:,0], train_data[:,1], train_data[:,2])
#     ax.axis(domain)
#     ax.set_zlim(zlim)
#     plt.title('Train data')

#     ax = fig.add_subplot(2,3,2, projection='3d')
#     ax.scatter(gAE_gen[:,0], gAE_gen[:,1], gAE_gen[:,2])
#     ax.axis(domain)
#     ax.set_zlim(zlim)
#     plt.title('gAE generation')

#     ax = fig.add_subplot(2,3,5, projection='3d')
#     ax.scatter(VAE_gen[:,0], VAE_gen[:,1], VAE_gen[:,2])
#     ax.axis(domain)
#     ax.set_zlim(zlim)
#     plt.title('VAE generation')

#     ax = fig.add_subplot(2,3,3, projection='3d')
#     ax.scatter(gAE_recon[:,0], gAE_recon[:,1], gAE_recon[:,2])
#     ax.axis(domain)
#     ax.set_zlim(zlim)
#     plt.title('gAE reconstruction')

#     ax = fig.add_subplot(2,3,6, projection='3d')
#     ax.scatter(VAE_recon[:,0], VAE_recon[:,1], VAE_recon[:,2])
#     ax.axis(domain)
#     ax.set_zlim(zlim)
#     plt.title('VAE reconstruction')

#     return fig

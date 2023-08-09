import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# from loss import log_t_normalizing_const
# from sampling import t_density, t_density_contour

def drawing(test_data, title_list, data_list, xmin, xmax, ymin, ymax, bins_x, bins_y,
            is_colorbar = False, wire_frame = True) : 

    edges_x = np.linspace(xmin, xmax, bins_x+1)
    edges_y = np.linspace(ymin, ymax, bins_y+1)  
    
    def log_tick_formatter(val, pos=None):
        return f"$10^{{{int(val)}}}$"
    
    M = len(data_list)
    fig = plt.figure(figsize = (3.5 * (M+1), 7))

    x = test_data[:,0].cpu().numpy()
    y = test_data[:,1].cpu().numpy() 
    ax = fig.add_subplot(2, M+1, 1, projection='3d')
    hist, *_ = np.histogram2d(x, y, bins=(edges_x,edges_y),density=True)
    mesh_X, mesh_Y = np.meshgrid(edges_x[1:],edges_y[1:])
    if wire_frame:
        surf = ax.plot_wireframe(mesh_X, mesh_Y, hist,rstride=1, cstride=1, linewidth=0.75)
    else:
        surf = ax.plot_surface(mesh_X, mesh_Y, hist, rstride=1, cstride=1, cmap='viridis',
                        linewidth=0, antialiased=False)
    if is_colorbar:
        axins = inset_axes(ax,
                    width="5%",  
                    height="50%",
                    loc='right',
                    borderpad=-5
                    )
        fig.colorbar(surf, cax=axins, orientation="vertical")
    ax.set_title('Test data',pad=-5)


    ax = fig.add_subplot(2, M+1, M+2, projection='3d')
    hist, *_ = np.histogram2d(x, y, bins=(edges_x,edges_y),density=True)
    eps = 0.1 / x.shape[0]
    hist = np.log10(hist+eps)

    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    mesh_X, mesh_Y = np.meshgrid(edges_x[1:],edges_y[1:])
    if wire_frame:
        surf = ax.plot_wireframe(mesh_X, mesh_Y, hist,rstride=1, cstride=1, linewidth=0.75)
    else:
        surf = ax.plot_surface(mesh_X, mesh_Y, hist, rstride=1, cstride=1, cmap='viridis',
                        linewidth=0, antialiased=False)
    if is_colorbar:
        axins = inset_axes(ax,
                    width="5%",  
                    height="50%",
                    loc='right',
                    borderpad=-5
                    )
        fig.colorbar(surf, cax=axins, orientation="vertical")
    ax.set_title("log scale",pad=-5)

    for m in range(M) : 
        x = data_list[m][:,0].cpu().numpy()
        y = data_list[m][:,1].cpu().numpy() 

        ax = fig.add_subplot(2, M+1, m+2, projection='3d')
        hist, *_ = np.histogram2d(x, y, bins=(edges_x,edges_y),density=True)
        mesh_X, mesh_Y = np.meshgrid(edges_x[1:],edges_y[1:])
        if wire_frame:
            surf = ax.plot_wireframe(mesh_X, mesh_Y, hist,rstride=1, cstride=1, linewidth=0.75)
        else:
            surf = ax.plot_surface(mesh_X, mesh_Y, hist, rstride=1, cstride=1, cmap='viridis',
                            linewidth=0, antialiased=False)
        if is_colorbar:
            axins = inset_axes(ax,
                        width="5%",  
                        height="50%",
                        loc='right',
                        borderpad=-5
                        )
            fig.colorbar(surf, cax=axins, orientation="vertical")
        ax.set_title(title_list[m],pad=-5)


        ax = fig.add_subplot(2, M+1, M+m+3, projection='3d')
        hist, *_ = np.histogram2d(x, y, bins=(edges_x,edges_y),density=True)
        eps = 0.1 / x.shape[0]
        hist = np.log10(hist+eps)

        ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        mesh_X, mesh_Y = np.meshgrid(edges_x[1:],edges_y[1:])
        if wire_frame:
            surf = ax.plot_wireframe(mesh_X, mesh_Y, hist,rstride=1, cstride=1, linewidth=0.75)
        else:
            surf = ax.plot_surface(mesh_X, mesh_Y, hist, rstride=1, cstride=1, cmap='viridis',
                            linewidth=0, antialiased=False)
        if is_colorbar:
            axins = inset_axes(ax,
                        width="5%",  
                        height="50%",
                        loc='right',
                        borderpad=-5
                        )
            fig.colorbar(surf, cax=axins, orientation="vertical")
        ax.set_title("log scale",pad=-5)
    return fig

def draw_3D_surfaces(data_point, x_range,y_range, bins
                    ,title, log_scale, is_colorbar = True, wire_frame=False):
    def log_tick_formatter(val, pos=None):
        return f"$10^{{{int(val)}}}$"
    nb_plot = len(data_point)
    fig = plt.figure(figsize=(3.5 * (nb_plot//2), 7))
    xmin, xmax = x_range
    ymin, ymax = y_range
    bins_x, bins_y = bins
    for i in range(nb_plot):
        ax = fig.add_subplot(nb_plot // 2, 2, i+1, projection='3d')
        
        # plot a 3D surface like in the example mplot3d/surface3d_demo
        
        edges_x = np.linspace(xmin, xmax, bins_x+1)
        edges_y = np.linspace(ymin, ymax, bins_y+1)
        x,y = data_point[i][:,0].numpy(),data_point[i][:,1].numpy() # data_point : [N, m_dim]


        hist, *_ = np.histogram2d(x, y, bins=(edges_x,edges_y),density=True)
        if log_scale:
            eps = 10e-10
            hist = np.log10(hist+eps)

            ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
            ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            
        mesh_X, mesh_Y = np.meshgrid(edges_x[1:],edges_y[1:])
        if wire_frame:
            surf = ax.plot_wireframe(mesh_X, mesh_Y, hist,rstride=1, cstride=1, linewidth=0.75)
        else:
            surf = ax.plot_surface(mesh_X, mesh_Y, hist, rstride=1, cstride=1, cmap='viridis',
                            linewidth=0, antialiased=False)
        if is_colorbar:
            axins = inset_axes(ax,
                        width="5%",  
                        height="50%",
                        loc='right',
                        borderpad=-5
                        )
            fig.colorbar(surf, cax=axins, orientation="vertical")
        ax.set_title(title[i],pad=-5)

        
    return fig



# def visualize_density(model_title_list, model_gen_list, 
#                       K, sample_nu_list, sample_mu_list, sample_var_list, ratio_list, xlim) :
#     model_gen_list = [gen[torch.isfinite(gen)].cpu().numpy() for gen in model_gen_list]

#     M = len(model_gen_list)
#     input = np.arange(-xlim * 100, xlim * 100 + 1) * 0.01
#     contour = t_density_contour(input, K, sample_nu_list, sample_mu_list, sample_var_list, ratio_list).squeeze().numpy()

#     # plot
#     fig = plt.figure(figsize = (3.5 * M, 7))

#     for m in range(M) : 
#         ax = fig.add_subplot(2,M,m+1)
#         plt.plot(input, contour, color='black')
#         plt.hist(model_gen_list[m], bins = 100, range = [-10, 10], density=True, alpha = 0.5, color='dodgerblue')
#         plt.xlim(-10, 10)
#         plt.title(f'{model_title_list[m]}')

#         ax = fig.add_subplot(2,M,M+m+1)
#         plt.plot(input, contour, color='black')
#         plt.hist(model_gen_list[m], bins = 100, range = [-xlim, xlim], density=True, alpha = 0.5, color='dodgerblue')
#         plt.xlim(-xlim, xlim)
#         plt.yscale("log")
#         plt.ylim(1e-6, 1)

#     return fig


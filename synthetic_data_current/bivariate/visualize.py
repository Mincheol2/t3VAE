import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as colors

from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# def drawing_rev(train_data, xmin, xmax, ymin, ymax, bins_x, bins_y,
#             is_colorbar = False, wire_frame = True) : 

#     edges_x = np.linspace(xmin, xmax, bins_x+1)
#     edges_y = np.linspace(ymin, ymax, bins_y+1)  
    
#     def log_tick_formatter(val, pos=None):
#         return f"$10^{{{int(val)}}}$"
    
#     fig = plt.figure(figsize = (14,8))
    
#     x = train_data[:,0].cpu().numpy()
#     y = train_data[:,1].cpu().numpy() 
#     ax = fig.add_subplot(1,2, 1, projection='3d')
#     hist, *_ = np.histogram2d(x, y, bins=(edges_x,edges_y),density=True)
#     hist = hist.transpose()
#     mesh_X, mesh_Y = np.meshgrid(edges_x[1:],edges_y[1:])
#     if wire_frame:
#         surf = ax.plot_wireframe(mesh_X, mesh_Y, hist,rstride=1, cstride=1, linewidth=0.75)
#     else:
#         surf = ax.plot_surface(mesh_X, mesh_Y, hist, rstride=1, cstride=1, cmap='viridis',
#                         linewidth=0, antialiased=False)
#     if is_colorbar:
#         axins = inset_axes(ax,
#                     width="5%",  
#                     height="50%",
#                     loc='right',
#                     borderpad=-5
#                     )
#         fig.colorbar(surf, cax=axins, orientation="vertical")
#     ax.set_title('Train data',pad=-5)


#     ax = fig.add_subplot(1,2,2, projection='3d')
#     hist, *_ = np.histogram2d(x, y, bins=(edges_x,edges_y),density=True)
#     hist = hist.transpose()
#     eps = 0.1 / x.shape[0]
#     hist = np.log10(hist+eps)

#     ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
#     ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
#     mesh_X, mesh_Y = np.meshgrid(edges_x[1:],edges_y[1:])
#     if wire_frame:
#         surf = ax.plot_wireframe(mesh_X, mesh_Y, hist,rstride=1, cstride=1, linewidth=0.75)
#     else:
#         surf = ax.plot_surface(mesh_X, mesh_Y, hist, rstride=1, cstride=1, cmap='viridis',
#                         linewidth=0, antialiased=False)
#     if is_colorbar:
#         axins = inset_axes(ax,
#                     width="5%",  
#                     height="50%",
#                     loc='right',
#                     borderpad=-5
#                     )
#         fig.colorbar(surf, cax=axins, orientation="vertical")
#     ax.set_title("log scale",pad=-5)
    
#     return fig

def drawing(test_data, title_list, data_list, xmin, xmax, ymin, ymax, bins, bins_x, bins_y,
            is_colorbar = False, wire_frame = True, angle = (25, 340)) : 
    
    colors_1 = plt.cm.Greys(0)
    colors_2 = plt.cm.terrain(np.linspace(0, 1, 256))
    all_colors = np.vstack((colors_1, colors_2))
    new_cmap = colors.LinearSegmentedColormap.from_list('viridis', all_colors)

    narrow_x = np.linspace(-10, 15, 30+1)
    narrow_y = np.linspace(-5, 5, 30+1)

    edges_x = np.linspace(xmin, xmax, bins_x+1)
    edges_y = np.linspace(ymin, ymax, bins_y+1)  
    
    def log_tick_formatter(val, pos=None):
        return f"$10^{{{int(val)}}}$"
    
    M = len(data_list)
    fig = plt.figure(figsize = (5.5 * (M+1), 21))
    x = test_data[:,0].cpu().numpy()
    y = test_data[:,1].cpu().numpy() 


    ax = fig.add_subplot(3, M+1, 1, projection='3d')
    hist, *_ = np.histogram2d(x, y, bins=(narrow_x,narrow_y),density=True)
    hist = hist.transpose()
    mesh_X, mesh_Y = np.meshgrid(narrow_x[1:],narrow_y[1:])
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
    ax.view_init(angle[0], angle[1])
    ax.set_title('Test data',pad=-5)


    ax = fig.add_subplot(3, M+1, M+2, projection='3d')
    hist, *_ = np.histogram2d(x, y, bins=(edges_x,edges_y),density=True)
    hist = hist.transpose()
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
    ax.view_init(angle[0], angle[1])
    ax.set_title("log scale",pad=-5)

    ax = fig.add_subplot(3, M+1, 2*M+3)
    hist, xedges,yedges = np.histogram2d(x,y, bins=bins,range=[[xmin, xmax],[ymin, ymax]],density=True)
    plt.imshow(hist, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],cmap = new_cmap)


    for m in range(M) : 
        x = data_list[m][:,0].cpu().numpy()
        y = data_list[m][:,1].cpu().numpy() 


        ax = fig.add_subplot(3, M+1, m+2, projection='3d')
        hist, *_ = np.histogram2d(x, y, bins=(narrow_x,narrow_y),density=True)
        hist = hist.transpose()
        mesh_X, mesh_Y = np.meshgrid(narrow_x[1:],narrow_y[1:])
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
        ax.view_init(angle[0], angle[1])
        ax.set_title(title_list[m],pad=-5)


        ax = fig.add_subplot(3, M+1, M+m+3, projection='3d')
        hist, *_ = np.histogram2d(x, y, bins=(edges_x,edges_y),density=True)
        hist = hist.transpose()
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
        ax.view_init(angle[0], angle[1])
        ax.set_title("log scale",pad=-5)

        ax = fig.add_subplot(3, M+1, 2*M+m+4)
        hist, xedges, yedges = np.histogram2d(x,y, bins=bins, range=[[xmin, xmax],[ymin, ymax]],density=True)
        ax.imshow(hist, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap = new_cmap)

    return fig

# def draw_3D_surfaces(data_point, x_range,y_range, bins
#                     ,title, log_scale, is_colorbar = True, wire_frame=False):
#     def log_tick_formatter(val, pos=None):
#         return f"$10^{{{int(val)}}}$"
#     nb_plot = len(data_point)
#     fig = plt.figure(figsize=(3.5 * (nb_plot//2), 7))
#     xmin, xmax = x_range
#     ymin, ymax = y_range
#     bins_x, bins_y = bins
#     for i in range(nb_plot):
#         ax = fig.add_subplot(nb_plot // 2, 2, i+1, projection='3d')
        
#         # plot a 3D surface like in the example mplot3d/surface3d_demo
        
#         edges_x = np.linspace(xmin, xmax, bins_x+1)
#         edges_y = np.linspace(ymin, ymax, bins_y+1)
#         x,y = data_point[i][:,0].numpy(),data_point[i][:,1].numpy() # data_point : [N, m_dim]


#         hist, *_ = np.histogram2d(x, y, bins=(edges_x,edges_y),density=True)
#         if log_scale:
#             eps = 10e-10
#             hist = np.log10(hist+eps)

#             ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
#             ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            
#         mesh_X, mesh_Y = np.meshgrid(edges_x[1:],edges_y[1:])
#         if wire_frame:
#             surf = ax.plot_wireframe(mesh_X, mesh_Y, hist,rstride=1, cstride=1, linewidth=0.75)
#         else:
#             surf = ax.plot_surface(mesh_X, mesh_Y, hist, rstride=1, cstride=1, cmap='viridis',
#                             linewidth=0, antialiased=False)
#         if is_colorbar:
#             axins = inset_axes(ax,
#                         width="5%",  
#                         height="50%",
#                         loc='right',
#                         borderpad=-5
#                         )
#             fig.colorbar(surf, cax=axins, orientation="vertical")
#         ax.set_title(title[i],pad=-5)

        
#     return fig




def draw_2D_heatmap(data_point, x_range,y_range, bins, title,per_fig_size=(3.5,7)):
    
    colors_1 = plt.cm.Greys(0)
    colors_2 = plt.cm.terrain(np.linspace(0, 1, 256))
    all_colors = np.vstack((colors_1, colors_2))
    new_cmap = colors.LinearSegmentedColormap.from_list('viridis', all_colors)
    nb_plot = len(data_point)
    num_row = nb_plot // 2 if nb_plot != 1 else 1

    
    fig = plt.figure(figsize=(per_fig_size[0] * num_row, per_fig_size[1]))
    for i in range(nb_plot):
        ax = fig.add_subplot(num_row, 2, i+1)
        
        # plot a 3D surface like in the example mplot3d/surface3d_demo
        
        x,y = data_point[i][:,0].numpy(),data_point[i][:,1].numpy() # data_point : [N, m_dim]


        hist, xedges,yedges = np.histogram2d(x,y, bins=bins,range=[x_range,y_range],density=True)

        
        ax.set_title(title[i],pad=-5)
        plt.imshow(hist, interpolation='nearest', origin='lower',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],cmap = new_cmap)
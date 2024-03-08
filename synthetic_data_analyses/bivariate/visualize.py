import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as colors

from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def draw_heatmap(test_data, data_list, title_list, x_range, y_range, bins, per_fig_size=(3.5,7)):
    test_data = test_data.cpu().numpy()
    data_list = [data.cpu().numpy() for data in data_list]
    
    all_colors = np.vstack((plt.cm.Greys(0), plt.cm.terrain(np.linspace(0, 1, 256))))
    new_cmap = colors.LinearSegmentedColormap.from_list('viridis', all_colors)
    nb_plot = len(data_list)
    num_row = (nb_plot+2) // 2 if nb_plot != 1 else 1

    fig = plt.figure(figsize=(per_fig_size[0] * num_row, per_fig_size[1]))

    ax = fig.add_subplot(2, num_row, 1)
    x, y = test_data[:,0], test_data[:,1] # data_point : [N, m_dim]
    hist, xedges,yedges = np.histogram2d(x,y, bins=bins,range=[x_range,y_range],density=True)
    ax.set_title('Test data',pad=-5)
    plt.imshow(hist.T, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],cmap = new_cmap)
    

    for i in range(nb_plot):
        ax = fig.add_subplot(2, num_row, i+2)
        
        x, y = data_list[i][:,0], data_list[i][:,1] # data_point : [N, m_dim]
        hist, xedges,yedges = np.histogram2d(x,y, bins=bins,range=[x_range,y_range],density=True)
        ax.set_title(title_list[i],pad=-5)
        plt.imshow(hist.T, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],cmap = new_cmap)
    return fig
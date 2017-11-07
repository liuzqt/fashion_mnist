# encoding: utf-8

'''

@author: ZiqiLiu


@file: plot.py

@time: 2017/11/5 下午1:17

@desc:
'''

import matplotlib

# matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt


def create_color_bar(f, cmap, max_epochs, title='Epochs'):
    colorbar_axis = [0.933, 0.125, 0.03, 0.83],
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar_ax = f.add_axes(rect=colorbar_axis)
    cbar = f.colorbar(sm, ticks=[], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(title, size=14)
    cbar.ax.text(0.5, -0.01, '0', transform=cbar.ax.transAxes,
                 va='top', ha='center', size=14)
    cbar.ax.text(0.5, 1.0, str(max_epochs), transform=cbar.ax.transAxes,
                 va='bottom', ha='center', size=14)


def plot_info_plain(IXT, ITY):
    '''
    
    :param IXT: of shape [batch_size,layers]
    :param ITY: of shape [batch_size,layers]
    :return: 
    '''
    cmap = plt.get_cmap('viridis')
    assert len(IXT) == len(ITY)

    f = plt.figure(figsize=(12, 10))

    colors = [cmap(i) for i in np.linspace(0, 1, len(IXT))]
    for ixt, ity, color in zip(IXT, ITY, colors):
        print(ixt,ity)
        assert len(ixt) == len(ity)
        plt.plot(ixt, ity, marker='o', ls='-', markersize=7,
                 markeredgewidth=0.04, linewidth=2.1, color=color)

    plt.xlabel("I(X;T)")
    plt.ylabel("I(T;Y)")
    create_color_bar(f, cmap, len(IXT))
    plt.show()
    plt.savefig('./fig.png')


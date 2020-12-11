#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:21:40 2020

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 18}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
data = np.load('results_noaa.npz')
plotting_stats = data['plotting_stats']

#%%
labels = [i+1 for i in range(plotting_stats.shape[0])]
all_data = [plotting_stats[i,1:] for i in range(plotting_stats.shape[0])]

mean =  [np.mean(plotting_stats[i,1:]) for i in range(plotting_stats.shape[0])]
median = [np.median(plotting_stats[i,1:]) for i in range(plotting_stats.shape[0])]
minimum = [np.min(plotting_stats[i,1:]) for i in range(plotting_stats.shape[0])]

#%%
red_square = dict(markerfacecolor='r', marker='s')

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

# rectangular box plot
bplot1 = axes.boxplot(all_data, sym='g*',
                      vert=True,  # vertical box alignment
                      patch_artist=False,  # fill with color
                      labels=labels, # will be used to label x-ticks
                      showfliers=True, showmeans=False)
                      #flierprops=red_square)  

axes.plot(labels,minimum,'ro-',lw=2,label='Best result')
axes.plot(labels,median,'bs-',lw=2,label='Median result')
#axes.plot(labels,mean,'r',lw=2)

#axes.set_title('Rectangular box plot')
axes.legend(loc=0)
axes.set_yscale('log')
#axes.set_ylim()
#axes.yaxis.grid(True)
axes.set_xlabel('Number of generations')
axes.set_ylabel('Mean squared error')

fig.tight_layout()
plt.show()
fig.savefig('boxplot.png',dpi=300)

#%%
k = 10
labels = [i+1 for i in range(k)]
all_data = [plotting_stats[i,1:] for i in range(k)]

mean =  [np.mean(plotting_stats[i,1:]) for i in range(k)]
median = [np.median(plotting_stats[i,1:]) for i in range(k)]
minimum = [np.min(plotting_stats[i,1:]) for i in range(k)]

#%%
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

# rectangular box plot
bplot1 = axes.boxplot(all_data, sym='g*',
                      vert=True,  # vertical box alignment
                      patch_artist=False,  # fill with color
                      labels=labels, # will be used to label x-ticks
                      showmeans=False)  

axes.plot(labels,minimum,'ro-',lw=2,label='Best result')
axes.plot(labels,median,'bs-',lw=2,label='Median result')
#axes.plot(labels,mean,'r',lw=2)

#axes.set_title('Rectangular box plot')
axes.legend(loc=0)
axes.set_yscale('log')
#axes.set_ylim()
#axes.yaxis.grid(True)
axes.set_xlabel('Number of generations')
axes.set_ylabel('Mean squared error')

fig.tight_layout()
plt.show()
fig.savefig('boxplot_small.png',dpi=300)

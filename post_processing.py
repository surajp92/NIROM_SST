#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:31:32 2020

@author: suraj
"""
import numpy as np

data = np.load('progress_19.npz')

population = data['population']
fitness = data['fitness']
population_old = data['population_old']

#%%
data = np.load('results_noaa.npz', allow_pickle=True)
best_param_dict = data['best_param_dict']

#%%
aa = best_param_dict[1][:]
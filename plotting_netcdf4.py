#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:32:51 2020

@author: suraj
"""

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

font = {'family' : 'Times New Roman',
        'size'   : 10}    
plt.rc('font', **font)


#'weight' : 'bold'

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
import h5py
from tqdm import tqdm as tqdm

f = h5py.File('./sst_weekly.mat','r') # can be downloaded from https://drive.google.com/drive/folders/1pVW4epkeHkT2WHZB7Dym5IURcfOP4cXu?usp=sharing
lat = np.array(f['lat'])
lon = np.array(f['lon'])
sst = np.array(f['sst'])
time = np.array(f['time'])

sst2 = np.zeros((len(sst[:,0]),len(lat[0,:]),len(lon[0,:])))
for i in tqdm(range(len(sst[:,0]))):
    sst2[i,:,:] = np.flipud((sst[i,:].reshape(len(lat[0,:]),len(lon[0,:]),order='F')))
    # sst2[i,:,:] = np.flipud((sst[i,:].reshape(len(lon[0,:]),len(lat[0,:]))))
    
#%%
lon1 = np.hstack((np.flip(-lon[0,:180]),lon[0,:180]))

x,y = np.meshgrid(lat,lon1,indexing='ij')


#%%
data = np.load('best_model_prediction.npz')
at = data['at']
ytest_ml = data['ytest_ml']
sst_masked_small_fluct = data['sst_masked_small_fluct'] 
sst_average_small = data['sst_average_small']
tfluc_pred = data['tfluc_pred']
Tpred = data['Tpred'] 
not_nan_array = data['not_nan_array']

#%%   
num_samples_train = 1914

Tpred_mid = np.zeros(not_nan_array.shape[0])
Tpred_final = np.zeros(not_nan_array.shape[0])

mid = int(num_samples_train/2)
final = num_samples_train - 1

Tpred_mid[Tpred_mid == 0] = 'nan'
Tpred_mid[not_nan_array] = Tpred[:,mid]

Tpred_final[Tpred_final == 0] = 'nan'
Tpred_final[not_nan_array] = Tpred[:,final]

trec_mid = np.flipud((Tpred_mid.reshape(len(lat[0,:]),len(lon[0,:]),order='F')))
trec_final = np.flipud((Tpred_final.reshape(len(lat[0,:]),len(lon[0,:]),order='F')))

trec_mid_shifted = np.hstack((trec_mid[:,180:],trec_mid[:,:180]))
trec_final_shifted = np.hstack((trec_final[:,180:],trec_final[:,:180]))

sst_mid = np.hstack((sst2[mid,:,180:],sst2[mid,:,:180]))
sst_final = np.hstack((sst2[final,:,180:],sst2[final,:,:180]))

#%%
fig = plt.figure(figsize=(10,6))

fig.add_subplot(2,2,1)

m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=-180,urcrnrlon=180,resolution='c')

m.drawcoastlines(linewidth=0.5)
m.fillcontinents(color='1.0')
m.drawmapboundary()
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,30.),linewidth=0.5,labels=[1,0,0,0])
m.drawmeridians(np.arange(-180.,181.,60.),linewidth=0.5,labels=[0,0,0,1])

x, y = m(*np.meshgrid(lon1,lat))
m.pcolormesh(x,y,sst_mid,vmin=0,vmax=30,shading='flat',cmap='jet')
m.colorbar(location='right')
plt.title("N = 750")

fig.add_subplot(2,2,2)

m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=-180,urcrnrlon=180,resolution='c')

m.drawcoastlines(linewidth=0.5)
m.fillcontinents(color='1.0')
m.drawmapboundary()
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,30.),linewidth=0.5,labels=[1,0,0,0])
m.drawmeridians(np.arange(-180.,181.,60.),linewidth=0.5,labels=[0,0,0,1])

x, y = m(*np.meshgrid(lon1,lat))
m.pcolormesh(x,y,sst_final,vmin=0,vmax=30,shading='flat',cmap='jet')
m.colorbar(location='right')
plt.title("N = 1500")

fig.add_subplot(2,2,3)

m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=-180,urcrnrlon=180,resolution='c')

m.drawcoastlines(linewidth=0.5)
m.fillcontinents(color='1.0')
m.drawmapboundary()
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,30.),linewidth=0.5,labels=[1,0,0,0])
m.drawmeridians(np.arange(-180.,181.,60.),linewidth=0.5,labels=[0,0,0,1])

x, y = m(*np.meshgrid(lon1,lat))
m.pcolormesh(x,y,trec_mid_shifted,vmin=0,vmax=30,shading='flat',cmap='jet')
m.colorbar(location='right')

fig.add_subplot(2,2,4)

m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=-180,urcrnrlon=180,resolution='c')

m.drawcoastlines(linewidth=0.5)
m.fillcontinents(color='1.0')
m.drawmapboundary()
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,30.),linewidth=0.5,labels=[1,0,0,0])
m.drawmeridians(np.arange(-180.,181.,60.),linewidth=0.5,labels=[0,0,0,1])

x, y = m(*np.meshgrid(lon1,lat))
m.pcolormesh(x,y,trec_final_shifted,vmin=0,vmax=30,shading='flat',cmap='jet')
m.colorbar(location='right')

fig.tight_layout()

plt.show()
fig.savefig('noaa_sst_true_pred.png',dpi=300)




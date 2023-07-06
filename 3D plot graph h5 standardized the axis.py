# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 18:03:13 2023

@author: limyu
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from scipy import interpolate
from scipy.signal import chirp, find_peaks, peak_widths
from scipy.stats import linregress
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import time


horizontal_peaks = []
horizontal_peaks_position = []
horizontal_peaks_max = []
horizontal_half = []
horizontal_full = []
horizontal_std_list = []

verticle_peaks = []
verticle_peaks_position = []
verticle_peaks_max = []
verticle_half = []
verticle_full = []
verticle_std_list = []

max_field_list = []

filename = ["grating012umpitch05dutycycle15um", "grating012umpitch05dutycycle20um", "grating12_11pitch2_8"]
link = ["C:\\Users\\limyu\\Downloads\grating012umpitch05dutycycle15um.h5", "C:\\Users\\limyu\\Downloads\grating012umpitch05dutycycle20um.h5"]

file = "grating012umpitch05dutycycle60um"
# Load the h5 file
with h5py.File("C:\\Users\\limyu\\Google Drive\\3d plots\\"+file+".h5", 'r') as f:
    # Get the dataset
    dset = f[file]
    # Load the dataset into a numpy array
    arr_3d_loaded = dset[()]



x = np.linspace(-20, 80, num=1950)
y = np.linspace(-25, 25, num = 975)
z = np.linspace(-5, 45, num = 317)

df_y_2d = arr_3d_loaded[:, int((arr_3d_loaded.shape[1]/2) - 0.5), :]
df_y_2d = df_y_2d.transpose()
colorbarmax = max(df_y_2d.max(axis=1))
X,Z = np.meshgrid(x,z)
fig = plt.figure(figsize=(18, 4))
ax = plt.axes()
cp=ax.contourf(X,Z,df_y_2d, 200, zdir='z', offset=-100, cmap='jet')
clb=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax, num=6), decimals=3)).tolist())
clb.ax.set_title('Electric Field (eV)', fontweight="bold")
for l in clb.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(15)
ax.set_xlabel('x-position (µm)', fontsize=15, fontweight="bold", labelpad=1)
ax.set_ylabel('y-position (µm)', fontsize=15, fontweight="bold", labelpad=1)


ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.show()
plt.close()

x = np.linspace(-20e-6, 80e-6, num=1950)
y = np.linspace(-25e-6, 25e-6, num = 975)
z = np.linspace(-5e-6, 45e-6, num = 317)       


N = np.arange(0, 317, 1)
for n in N:
    print(n, z[n])
    df = arr_3d_loaded[:,:,n]
    df = pd.DataFrame(df)
    df = df.transpose()
    max_E_field = df.max().max()
    row, col = np.where(df == max_E_field)
    row = int(float(row[0]))
    col = int(float(col[0]))
    
    max_field_list.append(max_E_field)

    hor_e = df.iloc[row, :]
    ver_e = df.iloc[:, col]
    
    #horizontal plot
    
    tck = interpolate.splrep(x, hor_e, s=0.0005, k=4) 
    x_new = np.linspace(min(x), max(x), 10000)
    y_fit = interpolate.BSpline(*tck)(x_new)
    peaks, _ = find_peaks(y_fit)
    peaks_h = x_new[peaks]
    peaks_height = y_fit[peaks]
    max_index = np.where(peaks_height == max(peaks_height))
    max_index = int(max_index[0])    
    
    horizontal_peaks.append(peaks_h)
    horizontal_peaks_position.append(x_new[np.where(y_fit == max(y_fit))[0][0]])
    horizontal_peaks_max.append(df.max().max())

    
    results_half = peak_widths(y_fit, peaks, rel_height=0.5)
    width = results_half[0]
    width = [i*(x_new[-1] - x_new[-2]) for i in width]
    width = np.array(width)
    height = results_half[1]
    x_min = results_half[2]
    x_min = np.array(x_new[np.around(x_min, decimals=0).astype(int)])
    x_max = results_half[3]
    x_max = np.array(x_new[np.around(x_max, decimals=0).astype(int)])    
    results_half_plot = (width, height, x_min, x_max)
    list_of_FWHM = results_half_plot[0]
    FWHM = list_of_FWHM[max_index]
    
    results_full = peak_widths(y_fit, peaks, rel_height=0.865)
    width_f = results_full[0]
    width_f = [i*(x_new[-1] - x_new[-2]) for i in width_f]
    width_f = np.array(width_f)
    height_f = results_full[1]
    x_min_f = results_full[2]
    x_min_f = np.array(x_new[np.around(x_min_f, decimals=0).astype(int)])
    x_max_f = results_full[3]
    x_max_f = np.array(x_new[np.around(x_max_f, decimals=0).astype(int)]) 
    results_full_plot = (width_f, height_f, x_min_f, x_max_f)
    list_of_waist = results_full_plot[0]
    waist = list_of_waist[max_index]
    A = max(y_fit)
    B = 1/(waist**2)
    def Gauss(x):
        y = A*np.exp(-1*B*x**2)
        return y
    max_E_index = np.where(y_fit == max(y_fit))
    max_E_index = int(max_E_index[0])
    x_max_E = x_new[max_E_index]
    distance = [x_max_E-i for i in x_new]
    distance = np.array(distance) 
    fit_y = Gauss(distance)
    std_dev = np.std(y_fit - fit_y)
    horizontal_std_list.append(std_dev)
    print("Height: ", z[n], "Horizontal Standard deviation: ", std_dev)
    
    horizontal_half.append(FWHM)      
    horizontal_full.append(waist)
    
    #vertical plot
    tck = interpolate.splrep(y, ver_e, s=0.0005, k=4) 
    x_new = np.linspace(min(y), max(y), 10000)
    y_fit = interpolate.BSpline(*tck)(x_new)
    peaks, _ = find_peaks(y_fit)
    peaks_v = x_new[peaks]
    peaks_height = y_fit[peaks]
    max_index = np.where(peaks_height == max(peaks_height))
    max_index = int(max_index[0])            
    
    verticle_peaks.append(peaks_v)
    verticle_peaks_position.append(x_new[np.where(y_fit == max(y_fit))[0][0]])
    verticle_peaks_max.append(df.max().max())
    
    results_half = peak_widths(y_fit, peaks, rel_height=0.5)
    width = results_half[0]
    width = [i*(x_new[-1] - x_new[-2]) for i in width]
    width = np.array(width)
    height = results_half[1]
    x_min = results_half[2]
    x_min = np.array(x_new[np.around(x_min, decimals=0).astype(int)])
    x_max = results_half[3]
    x_max = np.array(x_new[np.around(x_max, decimals=0).astype(int)])    
    results_half_plot = (width, height, x_min, x_max)
    list_of_FWHM = results_half_plot[0]
    FWHM = list_of_FWHM[max_index]
    
    results_full = peak_widths(y_fit, peaks, rel_height=0.865)
    width_f = results_full[0]
    width_f = [i*(x_new[-1] - x_new[-2]) for i in width_f]
    width_f = np.array(width_f)
    height_f = results_full[1]
    x_min_f = results_full[2]
    x_min_f = np.array(x_new[np.around(x_min_f, decimals=0).astype(int)])
    x_max_f = results_full[3]
    x_max_f = np.array(x_new[np.around(x_max_f, decimals=0).astype(int)]) 
    results_full_plot = (width_f, height_f, x_min_f, x_max_f)
    list_of_waist = results_full_plot[0]
    waist = list_of_waist[max_index]
    A = max(y_fit)
    B = 1/(waist**2)
    def Gauss(x):
        y = A*np.exp(-1*B*x**2)
        return y
    max_E_index = np.where(y_fit == max(y_fit))
    max_E_index = int(max_E_index[0])
    x_max_E = x_new[max_E_index]
    distance = [x_max_E-i for i in x_new]
    distance = np.array(distance) 
    fit_y = Gauss(distance)
    std_dev = np.std(y_fit - fit_y)
    verticle_std_list.append(std_dev)
    print("Height: ", z[n], "Vertical Standard deviation: ", std_dev)
    
    verticle_half.append(FWHM)      
    verticle_full.append(waist)

plt.plot(z, verticle_full)
plt.plot(z, horizontal_full)
plt.show()



from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.colors as colors



arr_3d_shortened = arr_3d_loaded[:, :, :]
arr_smaller = arr_3d_shortened[::20, ::20, ::20]
max_color = np.max(arr_smaller, axis=None)
min_color = np.min(arr_smaller, axis=None)

axis_0 = arr_smaller.shape[0]
axis_1 = arr_smaller.shape[1]
axis_2 = arr_smaller.shape[2]
arr_empty = np.empty((axis_0, axis_1, 0))



for i in range(axis_2):
    print(i)
    arr_2d = arr_smaller[:, :, i]
    max_val = arr_2d.max(axis=None)
    # Create a boolean mask for data points less than 0.02
    mask = arr_2d < 0.2*max_val
    
    # Set masked values to NaN
    arr_2d[mask] = np.nan
    
    arr_empty = np.concatenate((arr_empty, np.expand_dims(arr_2d, axis=2)), axis=2)




axis_0 = arr_empty.shape[0]
axis_1 = arr_empty.shape[1]
axis_2 = arr_empty.shape[2]

# Create a figure and axis
fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')
# Create x, y, and z indices
x = np.linspace(-20, 80, num=axis_0)
y = np.linspace(-25, 25, num = axis_1)
z = np.linspace(z[80]*1000000, 45, num = axis_2)


ax.view_init(elev=30, azim=210)
# Create a meshgrid of the indices
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

c = arr_empty.ravel()

# Plot the data as a 3D surface
cp = ax.scatter(xx, yy, zz, c=c, cmap='jet', norm=colors.LogNorm(), alpha=1)

ticks = (np.linspace(min_color, max_color, num = 6)).tolist()

# Add labels to the axis
clb=fig.colorbar(cp)

# Custom set the colorbar ticks
ticks = [0.01, 0.04, 0.08, 0.12, 0.16]
tick_labels = [str(i) for i in ticks]
clb.set_ticks(ticks)
clb.set_ticklabels(tick_labels)

ax.set_xlabel('x-position (µm)', fontsize=18, fontweight="bold", labelpad=13)
ax.set_ylabel('y-position (µm)', fontsize=18, fontweight="bold", labelpad=15)
ax.set_zlabel('z-position (µm)', fontsize=18, fontweight="bold", labelpad=15)

# set the x, y, and z axis ticks and bold the tick labels
ax.set_xticks([-20, 0.0, 20, 40, 60, 80])
ax.set_xticklabels(['-20', '0', '20', '40', '60', '80'], fontdict={'weight': 'bold'})
ax.set_yticks([-25, -15, -5, 5, 15, 25])
ax.set_yticklabels(['-25','-15', '-5', '5','15','25'], fontdict={'weight': 'bold'})
ax.set_zticks([10, 20, 30, 40])
ax.set_zticklabels(['10', '20', '30', '40'], fontdict={'weight': 'bold'})

ax.xaxis.label.set_fontsize(18)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(18)
ax.yaxis.label.set_weight("bold")
ax.zaxis.label.set_fontsize(18)
ax.zaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)


clb.ax.set_title('Electric field (eV)', fontweight="bold")
for l in clb.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(15)

# Show the plot
plt.show()




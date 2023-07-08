# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 20:36:52 2023

@author: limyu
"""

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
from sklearn.metrics import mean_squared_error
from scipy import interpolate
from scipy.signal import chirp, find_peaks, peak_widths
from scipy.stats import linregress
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import time

df_horizontal = pd.DataFrame()
df_verticle = pd.DataFrame()

horizontal_peaks = []
horizontal_positions = []
horizontal_peaks_position = []
horizontal_peaks_max = []
horizontal_half = []
horizontal_full = []
horizontal_mse_list = []

verticle_peaks = []
verticle_positions = []
verticle_peaks_position = []
verticle_peaks_max = []
verticle_half = []
verticle_full = []
verticle_mse_list = []

max_field_list = []

file_list = "grating12pitch100", "grating12_11pitch8_2", "grating12_11pitch6_4", "grating12_11pitch5_5", "grating12_11pitch4_6", "grating12_11pitch2_8" 

file = "grating12pitch100"
# Load the h5 file
with h5py.File("C:\\Users\\limyu\\Google Drive\\3d plots\\"+file+".h5", 'r') as f:
    # Get the dataset
    dset = f[file]
    # Load the dataset into a numpy array
    arr_3d_loaded = dset[()]

cmap = 'hot'

x = np.linspace(-20, 80, num=1950)
y = np.linspace(-25, 25, num = 975)
z = np.linspace(-5, 45, num = 317)

df_y_2d = arr_3d_loaded[:, int((arr_3d_loaded.shape[1]/2) - 0.5), :]
df_y_2d = df_y_2d.transpose()
colorbarmax = max(df_y_2d.max(axis=1))
X,Z = np.meshgrid(x,z)
fig = plt.figure(figsize=(18, 4))
ax = plt.axes()
cp=ax.contourf(X,Z,df_y_2d, 200, zdir='z', offset=-100, cmap=cmap)
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
    peaks, _ = find_peaks(hor_e)
    peaks_h = x[peaks].tolist()
    peaks_height = hor_e[peaks]
    max_index = np.where(peaks_height == max(peaks_height))
    max_index = int(max_index[0][0])   

    horizontal_positions.append(hor_e[peaks].to_list())
    horizontal_peaks.append(peaks_h)
    horizontal_peaks_position.append(x[np.where(hor_e == max(hor_e))[0][0]])
    horizontal_peaks_max.append(df.max().max())

    
    results_half = peak_widths(hor_e, peaks, rel_height=0.5)
    width = results_half[0]
    width = [i*(x[-1] - x[-2]) for i in width]
    width = np.array(width)
    height = results_half[1]
    x_min = results_half[2]
    x_min = np.array(x[np.around(x_min, decimals=0).astype(int)])
    x_max = results_half[3]
    x_max = np.array(x[np.around(x_max, decimals=0).astype(int)])    
    results_half_plot = (width, height, x_min, x_max)
    list_of_FWHM = results_half_plot[0]
    FWHM = list_of_FWHM[max_index]
    
    results_full = peak_widths(hor_e, peaks, rel_height=0.865)
    width_f = results_full[0]
    width_f = [i*(x[-1] - x[-2]) for i in width_f]
    width_f = np.array(width_f)
    height_f = results_full[1]
    x_min_f = results_full[2]
    x_min_f = np.array(x[np.around(x_min_f, decimals=0).astype(int)])
    x_max_f = results_full[3]
    x_max_f = np.array(x[np.around(x_max_f, decimals=0).astype(int)]) 
    results_full_plot = (width_f, height_f, x_min_f, x_max_f)
    list_of_waist = results_full_plot[0]
    waist = list_of_waist[max_index]
    A = max(hor_e)
    B = 1/(waist**2)
    def Gauss(x):
        y = A*np.exp(-1*B*x**2)
        return y
    max_E_index = np.where(hor_e == max(hor_e))
    max_E_index = int(max_E_index[0][0])
    x_max_E = x[max_E_index]
    distance = [i-x_max_E for i in x]
    distance = np.array(distance) 
    fit_y = Gauss(distance)
    mse_dev = mean_squared_error(hor_e, fit_y)
    horizontal_mse_list.append(mse_dev)
    print("Height: ", z[n], "Horizontal MSE: ", mse_dev)
   
    horizontal_half.append(FWHM)      
    horizontal_full.append(waist)
    
    #vertical plot 
    peaks, _ = find_peaks(ver_e)
    peaks_v = y[peaks].tolist()
    peaks_height = ver_e[peaks]
    max_index = np.where(peaks_height == max(peaks_height))
    max_index = int(max_index[0][0])            
    
    verticle_positions.append(ver_e[peaks].to_list())
    verticle_peaks.append(peaks_v)
    verticle_peaks_position.append(y[np.where(ver_e == max(ver_e))[0][0]])
    verticle_peaks_max.append(df.max().max())
    
    results_half = peak_widths(ver_e, peaks, rel_height=0.5)
    width = results_half[0]
    width = [i*(y[-1] - y[-2]) for i in width]
    width = np.array(width)
    height = results_half[1]
    x_min = results_half[2]
    x_min = np.array(y[np.around(x_min, decimals=0).astype(int)])
    x_max = results_half[3]
    x_max = np.array(y[np.around(x_max, decimals=0).astype(int)])    
    results_half_plot = (width, height, x_min, x_max)
    list_of_FWHM = results_half_plot[0]
    FWHM = list_of_FWHM[max_index]
    
    results_full = peak_widths(ver_e, peaks, rel_height=0.865)
    width_f = results_full[0]
    width_f = [i*(y[-1] - y[-2]) for i in width_f]
    width_f = np.array(width_f)
    height_f = results_full[1]
    x_min_f = results_full[2]
    x_min_f = np.array(y[np.around(x_min_f, decimals=0).astype(int)])
    x_max_f = results_full[3]
    x_max_f = np.array(y[np.around(x_max_f, decimals=0).astype(int)]) 
    results_full_plot = (width_f, height_f, x_min_f, x_max_f)
    list_of_waist = results_full_plot[0]
    waist = list_of_waist[max_index]
    A = max(ver_e)
    B = 1/(waist**2)
    def Gauss(x):
        y = A*np.exp(-1*B*x**2)
        return y
    max_E_index = np.where(ver_e == max(ver_e))
    max_E_index = int(max_E_index[0][0])
    x_max_E = y[max_E_index]
    distance = [i-x_max_E for i in y]
    distance = np.array(distance) 
    fit_y = Gauss(distance)
    mse_dev = mean_squared_error(y, fit_y)
    verticle_mse_list.append(mse_dev)
    print("Height: ", z[n], "Vertical MSE: ", mse_dev)

    verticle_half.append(FWHM)      
    verticle_full.append(waist)

verticle_full_plot = [i*1000000 for i in verticle_full]
horizontal_full_plot = [i*1000000 for i in horizontal_full]
s = 10
step = 1

import matplotlib.colors as colors

x = np.linspace(-20, 80, num=1950)
y = np.linspace(-25, 25, num = 975)
z = np.linspace(-5, 45, num = 317)

c = horizontal_mse_list
fig, ax = plt.subplots()
cp = ax.scatter(z[::step], horizontal_full_plot[::step], c=c[::step], cmap=cmap, norm=colors.LogNorm(), alpha=1, s=s)

clb=fig.colorbar(cp)

clb.ax.set_title('MSE', fontweight="bold")
for l in clb.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(15)
ax.set_xlabel('Height (z-axis, µm)', fontsize=15, fontweight="bold", labelpad=1)
ax.set_ylabel('Beam Waist along x-axis (µm)', fontsize=15, fontweight="bold", labelpad=1)
ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.axvline(x=7 , color='black', linestyle = "--")
ax.axvline(x=20 , color='black', linestyle = "--")
plt.show()
plt.close()

c = verticle_mse_list
fig, ax = plt.subplots()
cp = ax.scatter(z[::step], verticle_full_plot[::step], c=c[::step], cmap=cmap, norm=colors.LogNorm(), alpha=1, s=s)

clb=fig.colorbar(cp)

clb.ax.set_title('MSE', fontweight="bold")
for l in clb.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(15)
ax.set_xlabel('Height along z-axis (µm)', fontsize=15, fontweight="bold", labelpad=1)
ax.set_ylabel('Beam Waist along y-axis (µm)', fontsize=15, fontweight="bold", labelpad=1)
ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.axvline(x=7 , color='black', linestyle = "--")
ax.axvline(x=20 , color='black', linestyle = "--")
plt.show()
plt.close()



x = np.linspace(-20, 80, num=1950)
y = np.linspace(-25, 25, num = 975)
z = np.linspace(-5, 45, num = 317)

c = max_field_list
fig, ax = plt.subplots()
cp = ax.scatter(z[::step], horizontal_full_plot[::step], c=c[::step], cmap=cmap, norm=colors.LogNorm(), alpha=1, s=s)
clb=fig.colorbar(cp)

clb.ax.set_title('Electric Field (eV)', fontweight="bold")
for l in clb.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(15)
ax.set_xlabel('Height along z-axis (µm)', fontsize=15, fontweight="bold", labelpad=1)
ax.set_ylabel('Beam Waist along x-axis (µm)', fontsize=15, fontweight="bold", labelpad=1)
ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.axvline(x=7 , color='black', linestyle = "--")
ax.axvline(x=22 , color='black', linestyle = "--")
plt.show()
plt.close()



fig, ax = plt.subplots()
cp = ax.scatter(z[::step], verticle_full_plot[::step], c=c[::step], cmap=cmap, norm=colors.LogNorm(), alpha=1, s=s)

clb=fig.colorbar(cp)

clb.ax.set_title('Electric Field (eV)', fontweight="bold")
for l in clb.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(15)
ax.set_xlabel('Height along z-axis (µm)', fontsize=15, fontweight="bold", labelpad=1)
ax.set_ylabel('Beam Waist along y-axis (µm)', fontsize=15, fontweight="bold", labelpad=1)
ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.axvline(x=7 , color='black', linestyle = "--")
ax.axvline(x=20 , color='black', linestyle = "--")
plt.show()
plt.close()

z = np.linspace(-5e-6, 45e-6, num = 317)

df_horizontal["z"] = z
df_horizontal["max_field_list"] = max_field_list
df_horizontal["horizontal_peaks"] = horizontal_peaks
df_horizontal["horizontal_positions"] = horizontal_positions
df_horizontal["horizontal_peaks_position"] = horizontal_peaks_position
df_horizontal["horizontal_peaks_max"] = horizontal_peaks_max
df_horizontal["horizontal_half"] = horizontal_half
df_horizontal["horizontal_full"] = horizontal_full
df_horizontal["horizontal_mse_list"] = horizontal_mse_list

df_verticle["z"] = z
df_verticle["max_field_list"] = max_field_list
df_verticle["verticle_peaks"] = verticle_peaks
df_verticle["verticle_positions"] = verticle_positions
df_verticle["verticle_peaks_position"] = verticle_peaks_position
df_verticle["verticle_peaks_max"] = verticle_peaks_max
df_verticle["verticle_half"] = verticle_half
df_verticle["verticle_full"] = verticle_full
df_verticle["verticle_mse_list"] = verticle_mse_list

df_main = pd.concat([df_horizontal, df_verticle], axis= 1)

df_main.to_csv("C:\\Users\\limyu\\Google Drive\\3d plots\\"+file+".csv")

df_read = pd.read_csv("C:\\Users\\limyu\\Google Drive\\3d plots\\"+file+".csv",header=None, sep=",")


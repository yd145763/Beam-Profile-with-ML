# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 17:02:04 2023

@author: limyu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:24:26 2023

@author: limyu
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

df_main = pd.DataFrame()
R = 15,20,30,40,50,60
I = 0,1,2,3,4,5
for r, i in zip(R, I):
    url = "https://raw.githubusercontent.com/yd145763/Beam-Profile-with-ML/main/grating012umpitch05dutycycle"+str(r)+"um.csv"
    df = pd.read_csv(url)
    df = df.iloc[80:, :]
    
    def strip(s):
        return s.strip('[]')
    def split(s):
        return s.split(',')
        
    def float_list(str_list):
        return list(map(float, str_list))
    
    def count(s):
        return len(s)
    
    def std_dev(s):
        return np.std(s)
    
    df['horizontal_positions'] = df['horizontal_positions'].apply(strip)
    df['horizontal_positions'] = df['horizontal_positions'].apply(split)   
    df['horizontal_positions'] = df['horizontal_positions'].apply(float_list)
    
    df['horizontal_peaks'] = df['horizontal_peaks'].apply(strip)
    df['horizontal_peaks'] = df['horizontal_peaks'].apply(split)   
    df['horizontal_peaks'] = df['horizontal_peaks'].apply(float_list)
    
    df["horizontal_count"] = df['horizontal_positions'].apply(count)
    df["horizontal_std"] = df['horizontal_positions'].apply(std_dev)

    df['verticle_positions'] = df['verticle_positions'].apply(strip)
    df['verticle_positions'] = df['verticle_positions'].apply(split)   
    df['verticle_positions'] = df['verticle_positions'].apply(float_list)
    
    df['verticle_peaks'] = df['verticle_peaks'].apply(strip)
    df['verticle_peaks'] = df['verticle_peaks'].apply(split)   
    df['verticle_peaks'] = df['verticle_peaks'].apply(float_list)

    df["verticle_count"] = df['verticle_positions'].apply(count)
    df["verticle_std"] = df['verticle_positions'].apply(std_dev)

    df=df.assign(Radius=i)
    
    
    
    df_main = pd.concat([df_main, df], axis= 0)

print(df_main)

col_norm = ['z', 'max_field_list', 'horizontal_peaks_position',
            'horizontal_half', 'horizontal_full',
            'horizontal_mse_list',  'verticle_peaks_position', 
            'verticle_half', 'verticle_full', 'verticle_mse_list',
            'horizontal_count', 'horizontal_std', 'verticle_count', 'verticle_std']


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
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

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

    df=df.assign(Radius=r)
    
    
    
    df_main = pd.concat([df_main, df], axis= 0)

print(df_main)

col_norm = ['z', 'max_field_list', 'horizontal_peaks_position',
            'horizontal_half', 'horizontal_full',
            'horizontal_mse_list',  'Radius',
            'horizontal_count', 'horizontal_std']

df1_norm = df_main[col_norm]

# create a MinMaxScaler object
scaler = MinMaxScaler()
df2_norm = pd.DataFrame(scaler.fit_transform(df1_norm), columns=df1_norm.columns)
df2_norm["verticle_full"] = (df_main["verticle_full"]).to_list()
data = df2_norm 

# Split the data into training and test sets using Pandas
train_data = data.sample(frac=0.6, random_state=0)
test_data = data.drop(train_data.index)

# Separate the input variables (features) from the output variable (target) using Pandas
train_features = train_data.drop('verticle_full', axis=1)
train_labels = train_data['verticle_full']
test_features = test_data.drop('verticle_full', axis=1)
test_labels = test_data['verticle_full']

# Define the model using Keras
model = keras.Sequential([
    layers.Dense(1000, activation='relu', input_shape=[len(train_features.keys())]),
    layers.Dense(500, activation='relu'),
    layers.Dense(200, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(50, activation='relu'),
    layers.Dense(20, activation='relu'),
    layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(train_features, train_labels, epochs=100, validation_split=0.4)

# Evaluate the model
loss, mae = model.evaluate(test_features, test_labels)


# Print the results
print('Mean Absolute Error:', mae)

test_loss, test_acc = model.evaluate(test_features, test_labels, verbose=2)
results = model.evaluate(test_features, test_labels, verbose=2, batch_size=10)
print('\nTest accuracy:', test_acc)

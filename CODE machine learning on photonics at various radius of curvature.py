# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:52:45 2023

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


R = 15,20,30,40,50,60
I = 0,1,2,3,4,5
R_training = 15,20,40,50,60
I_training = 0,1,2,4,5
R_actual = 30
I_actual = 3

def get_df(R, I):
    df_name = pd.DataFrame()
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
        
        df['horizontal_half'] = [i*1000000 for i in df['horizontal_half']]
        df['verticle_half'] = [i*1000000 for i in df['verticle_half']]
        
        df_name = pd.concat([df_name, df], axis= 0)
    return df_name


df_main = get_df(R, I)
df_training = get_df(R_training, I_training)

col_feature = ['Radius', 'z', 'horizontal_count', 'horizontal_std', 'verticle_count', 'verticle_std']
df_main_fea = df_main[col_feature]
df_actual_fea = df_main_fea[df_main_fea['Radius'] ==30]

# create a MinMaxScaler object
scaler = MinMaxScaler()
df_main_fea_norm = pd.DataFrame(scaler.fit_transform(df_main_fea), columns=df_main_fea.columns)
df_actual_fea_norm =  df_main_fea_norm[df_main_fea_norm['Radius'] == df_main_fea_norm.iloc[500,0]]
df_training_fea_norm =  df_main_fea_norm[df_main_fea_norm['Radius'] != df_main_fea_norm.iloc[500,0]]


col_label = ['Radius', 'horizontal_full', 'verticle_full']
df_main_label = df_main[col_label]

df_main_label_norm = pd.DataFrame(scaler.fit_transform(df_main_label), columns=df_main_label.columns)
df_actual_label_norm =  df_main_label_norm[df_main_label_norm['Radius'] == df_main_label_norm.iloc[500,0]]
df_training_label_norm =  df_main_label_norm[df_main_label_norm['Radius'] != df_main_label_norm.iloc[500,0]]
df_training_label_norm = df_training_label_norm.iloc[:, 1:]


X = df_training_fea_norm
y = df_training_label_norm

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# Define the model using Keras
model = keras.Sequential([
    layers.Dense(200, activation='relu', input_shape=[len(X_train.keys())]),
    layers.Dense(150, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(80, activation='relu'),
    layers.Dense(50, activation='relu'),
    layers.Dense(30, activation='relu'),
    layers.Dense(20, activation='relu'),
    layers.Dense(10, activation='relu'),
    layers.Dense(5, activation='relu'),
    layers.Dense(y_train.shape[1])
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.3)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)

# Print the results
print('Mean Absolute Error:', mae)

# Define the model using Keras
model2 = keras.Sequential([
    layers.Dense(200, activation='elu', input_shape=[len(X_train.keys())]),
    layers.Dense(150, activation='elu'),
    layers.Dense(100, activation='elu'),
    layers.Dense(80, activation='elu'),
    layers.Dense(50, activation='elu'),
    layers.Dense(30, activation='elu'),
    layers.Dense(20, activation='elu'),
    layers.Dense(10, activation='elu'),
    layers.Dense(5, activation='elu'),
    layers.Dense(y_train.shape[1])
])

# Compile the model
model2.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model2.fit(X_train, y_train, epochs=100, validation_split=0.3)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)

# Print the results
print('Mean Absolute Error:', mae)


X_predict = df_actual_fea_norm

predictions2 = model2.predict(X_predict)

print(predictions2[:,0])
z = [i*1000000 for i in df_actual_fea['z']]
from matplotlib.ticker import StrMethodFormatter
fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
ax.scatter(z, df_actual_label_norm['horizontal_full'])
ax.scatter(z, predictions2[:,0])
#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Height from grating (µm)")
plt.ylabel("Normalized Horizontal Beam Waist")
plt.legend(["Original Data", "Predicted Data"], prop={'weight': 'bold','size': 10}, loc = "upper left")
plt.title("Normalized Horizontal Beam Waist", fontweight = 'bold')
plt.show()
plt.close()

predictions = model.predict(X_predict)

fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
ax.scatter(z, df_actual_label_norm['verticle_full'])
ax.scatter(z, predictions[:,1])
#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Height from grating (µm)")
plt.ylabel("Normalized Vertical Beam Waist")
plt.legend(["Original Data", "Predicted Data"], prop={'weight': 'bold','size': 10}, loc = "upper left")
plt.title("Normalized Vertical Beam Waist", fontweight = 'bold')
plt.show()
plt.close()

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
R = 15,20,40,50,60
I = 0,1,2,4,5
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
    
    df_main = pd.concat([df_main, df], axis= 0)

print(df_main.columns)

col_feature = ['Radius', 'z', 'horizontal_count', 'horizontal_std', 'verticle_count', 'verticle_std']

df1_norm = df_main[col_feature]

# create a MinMaxScaler object
scaler = MinMaxScaler()
#df2_norm = pd.DataFrame(scaler.fit_transform(df1_norm), columns=df1_norm.columns)
df_feature = df1_norm 

col_label = ['horizontal_full', 'verticle_full']

df_label = df_main[col_label]

X = df_feature
y = df_label

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



url_predict = "https://raw.githubusercontent.com/yd145763/Beam-Profile-with-ML/main/grating012umpitch05dutycycle30um.csv"
df_predict = pd.read_csv(url_predict)
df_predict = df_predict.iloc[80:, :]
df_predict=df_predict.assign(Radius=30)
df_predict['horizontal_positions'] = df_predict['horizontal_positions'].apply(strip)
df_predict['horizontal_positions'] = df_predict['horizontal_positions'].apply(split)   
df_predict['horizontal_positions'] = df_predict['horizontal_positions'].apply(float_list)

df_predict['horizontal_peaks'] = df_predict['horizontal_peaks'].apply(strip)
df_predict['horizontal_peaks'] = df_predict['horizontal_peaks'].apply(split)   
df_predict['horizontal_peaks'] = df_predict['horizontal_peaks'].apply(float_list)

df_predict["horizontal_count"] = df_predict['horizontal_positions'].apply(count)
df_predict["horizontal_std"] = df_predict['horizontal_positions'].apply(std_dev)

df_predict['verticle_positions'] = df_predict['verticle_positions'].apply(strip)
df_predict['verticle_positions'] = df_predict['verticle_positions'].apply(split)   
df_predict['verticle_positions'] = df_predict['verticle_positions'].apply(float_list)

df_predict['verticle_peaks'] = df_predict['verticle_peaks'].apply(strip)
df_predict['verticle_peaks'] = df_predict['verticle_peaks'].apply(split)   
df_predict['verticle_peaks'] = df_predict['verticle_peaks'].apply(float_list)

df_predict["verticle_count"] = df_predict['verticle_positions'].apply(count)
df_predict["verticle_std"] = df_predict['verticle_positions'].apply(std_dev)
df_predict['horizontal_half'] = [i*1000000 for i in df_predict['horizontal_half']]
df_predict['verticle_half'] = [i*1000000 for i in df_predict['verticle_half']]

column_predict = col_feature
X_predict = df_predict[column_predict]
#X_predict = pd.DataFrame(scaler.fit_transform(X_predict), columns=X_predict.columns)

predictions2 = model2.predict(X_predict)

print(predictions2[:,0])
z = [i*1000000 for i in df_predict['z']]
from matplotlib.ticker import StrMethodFormatter
fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
ax.scatter(z, df_predict['horizontal_full'])
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
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Height from grating (µm)")
plt.ylabel("Beam Waist")
plt.legend(["Original Data", "Predicted Data"], prop={'weight': 'bold','size': 10}, loc = "upper left")
plt.title("Horizontal Beam Waist", fontweight = 'bold')
plt.show()
plt.close()

predictions = model.predict(X_predict)

fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
ax.scatter(z, df_predict['verticle_full'])
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
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Height from grating (µm)")
plt.ylabel("Beam Waist")
plt.legend(["Original Data", "Predicted Data"], prop={'weight': 'bold','size': 10}, loc = "upper left")
plt.title("Vertical Beam Waist", fontweight = 'bold')
plt.show()
plt.close()

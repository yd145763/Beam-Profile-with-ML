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
from tensorflow.keras.callbacks import TensorBoard


R = pd.Series([15,20,30,40,50,60])
I = pd.Series(np.arange(0, len(R), 1))
R_actual = 30
I_actual = R.index[R == R_actual][0]
R_training = R[R!=R_actual]
I_training = I[I!=I_actual]
R_actual_norm = round((R_actual - min(R))/(max(R)-min(R)),5)


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
max_ver_full = max(df_main["verticle_full"])
min_ver_full = min(df_main["verticle_full"])
max_hor_full = max(df_main["horizontal_full"])
min_hor_full = min(df_main["horizontal_full"])
max_max_field_list = max(df_main["max_field_list"])
min_max_field_list = min(df_main["max_field_list"])


print(df_main.columns)
df_training = get_df(R_training, I_training)

col_feature = ['Radius', 'z']
df_main_fea = df_main[col_feature]
df_actual_fea = df_main_fea[df_main_fea['Radius'] ==R_actual]

# create a MinMaxScaler object
scaler = MinMaxScaler()
df_main_fea_norm = pd.DataFrame(scaler.fit_transform(df_main_fea), columns=df_main_fea.columns)
df_main_fea_norm['Radius']=df_main_fea_norm['Radius'].round(5)
df_actual_fea_norm =  df_main_fea_norm[df_main_fea_norm['Radius'] == R_actual_norm]
df_training_fea_norm =  df_main_fea_norm[df_main_fea_norm['Radius'] != R_actual_norm]


col_label = ['Radius', 'horizontal_full', 'verticle_full', 'max_field_list']
df_main_label = df_main[col_label]

df_main_label_norm = pd.DataFrame(scaler.fit_transform(df_main_label), columns=df_main_label.columns)
df_main_label_norm['Radius']=df_main_label_norm['Radius'].round(5)
df_actual_label_norm =  df_main_label_norm[df_main_label_norm['Radius'] == R_actual_norm]
df_training_label_norm =  df_main_label_norm[df_main_label_norm['Radius'] != R_actual_norm]
df_training_label_norm = df_training_label_norm.iloc[:, 1:]


X = df_training_fea_norm
y = df_training_label_norm

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

NAME1 = "Model 1"

tensorboard1 = TensorBoard(log_dir="logs/{}".format(NAME1))

# Define the model using Keras
model = keras.Sequential([
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
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

# Train the model
history = model.fit(X_train, y_train, epochs=200, validation_split=0.3, callbacks=[tensorboard1])

# Evaluate the model
loss1, mae1 = model.evaluate(X_test, y_test)

# Print the results
print('Mean Absolute Error:', mae1)




X_predict = df_actual_fea_norm

predictions = model.predict(X_predict)

pred_hor_full_denorm = [(i*(max_hor_full-min_hor_full))+min_hor_full for i in predictions[:,0]]
actual_hor_full_denorm = df_main[df_main["Radius"] == R_actual]["horizontal_full"]

pred_ver_full_denorm = [(i*(max_ver_full-min_ver_full))+min_ver_full for i in predictions[:,1]]
actual_ver_full_denorm = df_main[df_main["Radius"] == R_actual]["verticle_full"]

pred_max_field_list_denorm = [(i*(max_max_field_list-min_max_field_list))+min_max_field_list for i in predictions[:,2]]
actual_max_field_list_denorm = df_main[df_main["Radius"] == R_actual]["max_field_list"]


z = [i*1000000 for i in df_actual_fea['z']]

from matplotlib.ticker import StrMethodFormatter

fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
ax.scatter(z, actual_hor_full_denorm, s=1, color = "blue")
ax.plot(z, pred_hor_full_denorm, color = "red")
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
plt.ylabel("Horizontal Beam Waist")
plt.legend(["Original Data", "Predicted Data"], prop={'weight': 'bold','size': 10}, loc = "upper left")
plt.title("Horizontal Beam Waist", fontweight = 'bold')
plt.show()
plt.close()

fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
ax.scatter(z, actual_ver_full_denorm, s=1, color = "blue")
ax.plot(z, pred_ver_full_denorm, color = "red")
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
plt.ylabel("Vertical Beam Waist")
plt.legend(["Original Data", "Predicted Data"], prop={'weight': 'bold','size': 10}, loc = "upper left")
plt.title("Vertical Beam Waist", fontweight = 'bold')
plt.show()
plt.close()

fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
ax.scatter(z, actual_max_field_list_denorm, s=1, color = "blue")
ax.plot(z, pred_max_field_list_denorm, color = "red")
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
plt.ylabel("Max E-field (eV)")
plt.legend(["Original Data", "Predicted Data"], prop={'weight': 'bold','size': 10}, loc = "upper left")
plt.title("Max E-field", fontweight = 'bold')
plt.show()
plt.close()

fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
ax.scatter(z, df_actual_label_norm['horizontal_full'], s=1, color = "blue")
ax.plot(z, predictions[:,0], color = "red")
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
plt.ylabel("Horizontal Beam Waist\n(Normalized)")
plt.legend(["Original Data", "Predicted Data"], prop={'weight': 'bold','size': 10}, loc = "upper left")
plt.title("Normalized Horizontal Beam Waist", fontweight = 'bold')
plt.show()
plt.close()




fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
ax.scatter(z, df_actual_label_norm['verticle_full'], s=1, color = "blue")
ax.plot(z, predictions[:,1], color = "red")
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
plt.ylabel("Vertical Beam Waist\n(Normalized)")
plt.legend(["Original Data", "Predicted Data"], prop={'weight': 'bold','size': 10}, loc = "upper left")
plt.title("Normalized Vertical Beam Waist", fontweight = 'bold')
plt.show()
plt.close()


fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
ax.scatter(z, df_actual_label_norm['max_field_list'], s=1, color = "blue")
ax.plot(z, predictions[:,2], color = "red")
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
plt.ylabel("Max E-field\n(Normalized)")
plt.legend(["Original Data", "Predicted Data"], prop={'weight': 'bold','size': 10}, loc = "upper left")
plt.title("Normalized Max E-field", fontweight = 'bold')
plt.show()
plt.close()
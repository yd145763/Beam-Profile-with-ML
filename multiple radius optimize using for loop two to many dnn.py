# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:31:17 2023

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
from sklearn.metrics import mean_absolute_error
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from matplotlib.ticker import StrMethodFormatter

R = pd.Series([15,20,30,40,50,60])
I = pd.Series(np.arange(0, len(R), 1))
R_actual = pd.Series([15, 30, 50]) #select which radius you want to "remove" here
I_actual = R.index.where(R.isin(R_actual)).dropna()
R_training = R[~R.isin(R_actual)]
I_training = I[~I.isin(I_actual)]
R_actual_norm = [round((i - min(R))/(max(R)-min(R)), 5) for i in R_actual]
R_actual_norm = pd.Series(R_actual_norm)
R_index = 1 #the index where you want to predict from "R_actual"

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
df_actual_fea = df_main_fea[df_main_fea['Radius'].isin(R_actual)]

# create a MinMaxScaler object
scaler = MinMaxScaler()
df_main_fea_norm = pd.DataFrame(scaler.fit_transform(df_main_fea), columns=df_main_fea.columns)
df_main_fea_norm['Radius']=df_main_fea_norm['Radius'].round(5)
df_actual_fea_norm =  df_main_fea_norm[df_main_fea_norm['Radius'].isin(R_actual_norm)]
df_training_fea_norm =  df_main_fea_norm[~df_main_fea_norm['Radius'].isin(R_actual_norm)]


col_label = ['Radius', 'horizontal_full', 'verticle_full', 'max_field_list',
'horizontal_count', 'horizontal_std', 'verticle_count', 'verticle_std']
df_main_label = df_main[col_label]

df_main_label_norm = pd.DataFrame(scaler.fit_transform(df_main_label), columns=df_main_label.columns)
df_main_label_norm['Radius']=df_main_label_norm['Radius'].round(5)
df_actual_label_norm =  df_main_label_norm[df_main_label_norm['Radius'].isin(R_actual_norm)]
df_training_label_norm =  df_main_label_norm[~df_main_label_norm['Radius'].isin(R_actual_norm)]
df_training_label_norm = df_training_label_norm.iloc[:, 1:]

X_train_unweight = df_training_fea_norm
y_train = df_training_label_norm
X_test = df_actual_fea_norm
y_test = df_actual_label_norm.drop('Radius', axis =1)


feature_weights = [1, 1]
X_train = X_train_unweight * feature_weights

name = []
mae = []
nodes = []
layers1=[]
ape_actual_hor = []
ape_actual_ver = []
ape_actual_max_field_list = []
train_test_ape = []
dense_layers = [1,2,4,5,6,8,10,12]
layer_sizes = [4,8,10,20,50,150,250]
num_epochs = 100
s = 10

max_ver_full = max(df_main["verticle_full"])
min_ver_full = min(df_main["verticle_full"])
max_hor_full = max(df_main["horizontal_full"])
min_hor_full = min(df_main["horizontal_full"])
max_max_field_list = max(df_main["max_field_list"])
min_max_field_list = min(df_main["max_field_list"])
max_horizontal_count = max(df_main["horizontal_count"])
min_horizontal_count = min(df_main["horizontal_count"])
max_horizontal_std = max(df_main["horizontal_std"])
min_horizontal_std = min(df_main["horizontal_std"])
max_verticle_count = max(df_main["verticle_count"])
min_verticle_count = min(df_main["verticle_count"])
max_verticle_std = max(df_main["verticle_std"])
min_verticle_std = min(df_main["verticle_std"])

for dense_layer in dense_layers:
    for layer_size in layer_sizes:

        NAME = "{}-nodes-{}-dense-{}".format(layer_size, dense_layer, int(time.time()))
        print(NAME)
        name.append(NAME)
        nodes.append(layer_size)
        layers1.append(dense_layer)

        tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
        model = Sequential()
        model.add(Dense(len(X_train.keys()),  input_shape=[len(X_train.keys())]))
        for _ in range(dense_layer):
            if layer_size <= 5:
                break
            else:
                model.add(Dense(layer_size))
                model.add(Activation('elu'))
                #layer_size = int(round(layer_size*0.9, 0))

        model.add(Dense(y_train.shape[1]))


        # Compile the model
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
        
        # Train the model
        history = model.fit(X_train, y_train, epochs=num_epochs,validation_data=(X_test, y_test), callbacks=[tensorboard], batch_size = 10)
        
        # Evaluate the model
        loss1, mae1 = model.evaluate(X_test, y_test)
        
        # Print the results
        print('Mean Absolute Error:', mae1)
        mae.append(mae1)
        
        training_loss = pd.Series(history.history['loss'])
        validation_loss = pd.Series(history.history['val_loss'])
        
        diff = (validation_loss - training_loss)
        rel_error = diff / training_loss
        pct_error = rel_error * 100
        ape = pct_error.mean()
        train_test_ape.append(ape)
        
        epochs = range(1, num_epochs + 1)
        
        fig = plt.figure(figsize=(7, 4))
        ax = plt.axes()
        ax.plot(epochs, training_loss, color = "blue")
        ax.plot(epochs, validation_loss, color = "red")
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
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(["Training_loss", "Validation Loss"], prop={'weight': 'bold','size': 10}, loc = "upper left")
        plt.title("Training/Validation Loss\n"+NAME, fontweight = 'bold')
        plt.show()
        plt.close()
        
        X_predict = df_actual_fea_norm[df_actual_fea_norm["Radius"] == R_actual_norm[R_index]]
        
        predictions = model.predict(X_predict)
        
        pred_hor_full_denorm = [(i*(max_hor_full-min_hor_full))+min_hor_full for i in predictions[:,0]]
        actual_hor_full_denorm = df_main[df_main["Radius"] == R_actual[R_index]]["horizontal_full"]
        
        pred_ver_full_denorm = [(i*(max_ver_full-min_ver_full))+min_ver_full for i in predictions[:,1]]
        actual_ver_full_denorm = df_main[df_main["Radius"] == R_actual[R_index]]["verticle_full"]
        
        pred_max_field_list_denorm = [(i*(max_max_field_list-min_max_field_list))+min_max_field_list for i in predictions[:,2]]
        actual_max_field_list_denorm = df_main[df_main["Radius"] == R_actual[R_index]]["max_field_list"]
        
        df_actual_fea_filtered = df_actual_fea[df_actual_fea["Radius"]==R_actual[R_index]]
        z = [i*1000000 for i in df_actual_fea_filtered['z']]
        z = pd.Series(z)

        diff = (actual_hor_full_denorm - pred_hor_full_denorm).abs()
        rel_error = diff / actual_hor_full_denorm
        pct_error = rel_error * 100
        ape = pct_error.mean()
        ape_actual_hor.append(ape)
        
        fig = plt.figure(figsize=(7, 4))
        ax = plt.axes()
        ax.scatter(z, actual_hor_full_denorm, s=s, facecolor='none', edgecolor='blue')
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
        plt.title("Horizontal Beam Waist\n"+NAME, fontweight = 'bold')
        plt.show()
        plt.close()
        print(" ")
        print(" ")
        print(" ")
        print("Mean Absolute Error Horizontal Beam Waist", mean_absolute_error(actual_hor_full_denorm, pred_hor_full_denorm))
        print(" ")
        print(" ")
        print(" ")
        
        diff = (actual_ver_full_denorm - pred_ver_full_denorm).abs()
        rel_error = diff / actual_ver_full_denorm
        pct_error = rel_error * 100
        ape = pct_error.mean()
        ape_actual_ver.append(ape)
        
        fig = plt.figure(figsize=(7, 4))
        ax = plt.axes()
        ax.scatter(z, actual_ver_full_denorm, s=s,  facecolor='none', edgecolor='blue')
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
        plt.title("Vertical Beam Waist\n"+NAME+"\n", fontweight = 'bold')
        plt.show()
        plt.close()
        print(" ")
        print(" ")
        print(" ")
        print("Mean Absolute Error Vertical Beam Waist", mean_absolute_error(actual_ver_full_denorm, pred_ver_full_denorm))
        print(" ")
        print(" ")
        print(" ")
        
        
        """
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
        print(" ")
        print(" ")
        print(" ")
        print("Mean Absolute Error Max E-Field", mean_absolute_error(actual_max_field_list_denorm, pred_max_field_list_denorm))
        print(" ")
        print(" ")
        print(" ")
        
        """
        
df_mae = pd.DataFrame()
df_mae["name"] = name
df_mae["mae"] = mae
df_mae["number of layers"] = layers1
df_mae["number of nodes first layer"] = nodes
df_mae["ape_actual_ver"] = ape_actual_ver
df_mae["ape_actual_hor"] = ape_actual_hor
df_mae["train_test_ape"] = train_test_ape

plt.plot(df_mae["mae"])
plt.show()

import seaborn as sns
mat = df_mae.pivot('number of layers', 'number of nodes first layer', 'ape_actual_ver')
fig, ax = plt.subplots()
ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold", fontsize = 12)
ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold", fontsize = 12)
sns.heatmap(mat, annot=True, cmap='hot', annot_kws={"fontsize":10, "fontweight":"bold"}, fmt=".1f")
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
cbar.ax.set_title("Average\nPercentage\nError (%)", fontweight="bold")
font = {'color': 'black', 'weight': 'bold', 'size': 12}
ax.set_ylabel("Number of Layers, L", fontdict=font)
ax.set_xlabel("Number of Nodes, N\n(First Layer)", fontdict=font)
ax.tick_params(axis='both', labelsize=12, weight='bold')


mat = df_mae.pivot('number of layers', 'number of nodes first layer', 'ape_actual_hor')
fig, ax = plt.subplots()
ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold", fontsize = 12)
ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold", fontsize = 12)
sns.heatmap(mat, annot=True, cmap='hot', annot_kws={"fontsize":10, "fontweight":"bold"}, fmt=".1f")
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
cbar.ax.set_title("Average\nPercentage\nError (%)", fontweight="bold")
font = {'color': 'black', 'weight': 'bold', 'size': 12}
ax.set_ylabel("Number of Layers, L", fontdict=font)
ax.set_xlabel("Number of Nodes, N\n(First Layer)", fontdict=font)
ax.tick_params(axis='both', labelsize=12, weight='bold')

mat = df_mae.pivot('number of layers', 'number of nodes first layer', 'train_test_ape')
fig, ax = plt.subplots()
ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold", fontsize = 12)
ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold", fontsize = 12)
sns.heatmap(mat, annot=True, cmap='hot', annot_kws={"fontsize":10, "fontweight":"bold"}, fmt=".1f")
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
cbar.ax.set_title("Average\nPercentage\nError (%)", fontweight="bold")
font = {'color': 'black', 'weight': 'bold', 'size': 12}
ax.set_ylabel("Number of Layers, L", fontdict=font)
ax.set_xlabel("Number of Nodes, N\n(First Layer)", fontdict=font)
ax.tick_params(axis='both', labelsize=12, weight='bold')







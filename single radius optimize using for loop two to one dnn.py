# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:30:15 2023

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
R_actual = 30 #set the radius to be "removed" here
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
print(df_main.columns)

labels = ['max_field_list', 'verticle_half']

for label in labels:

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
    
    
    col_label = ['Radius', label]
    df_main_label = df_main[col_label]
    
    df_main_label_norm = pd.DataFrame(scaler.fit_transform(df_main_label), columns=df_main_label.columns)
    df_main_label_norm['Radius']=df_main_label_norm['Radius'].round(5)
    df_actual_label_norm =  df_main_label_norm[df_main_label_norm['Radius'] == R_actual_norm]
    df_training_label_norm =  df_main_label_norm[df_main_label_norm['Radius'] != R_actual_norm]
    df_training_label_norm = df_training_label_norm.iloc[:, 1:]
    
    
    X_train_unweight = df_training_fea_norm
    y_train = df_training_label_norm
    X_test = df_actual_fea_norm
    y_test = df_actual_label_norm
    

    feature_weights = [1, 1]
    X_train = X_train_unweight * feature_weights
    
    name = []
    mae = []
    nodes = []
    layers1=[]
    ape_list = []
    train_test_ape = []
    RMS = []
    dense_layers = [1,2,3,4,5,6,8]
    layer_sizes = [2,4,6,8,10,15,20]
    num_epochs = 100
    s = 10
    
    
    max_label = max(df_main[label])
    min_label = min(df_main[label])
    
    
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
                    model.add(Dense(layer_size))
                    model.add(Activation('elu'))
                    #layer_size = int(round(layer_size*0.9, 0))
    
            model.add(Dense(y_train.shape[1]))
    
    
            # Compile the model
            model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
            
            # Train the model
            history = model.fit(X_train, y_train, epochs=num_epochs,validation_data=(X_test, y_test), batch_size = 10)
            
            
            # Evaluate the model
            loss1, mae1 = model.evaluate(X_test, y_test)
            loss2, mae2 = model.evaluate(X_train, y_train)
            
            # Print the results
            print('Mean Absolute Error:', mae1)
            print('Training Loss', loss2)
            print('Validation loss', loss1)
            print(history.history['loss'])
            mae.append(mae1)

            training_loss = pd.Series(history.history['loss'])
            validation_loss = pd.Series(history.history['val_loss'])
            rms = np.sqrt(np.mean(validation_loss[50:] ** 2))
            RMS.append(rms)
            
            diff = (validation_loss - training_loss).abs()
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

            
            X_predict = df_actual_fea_norm
    
            predictions = model.predict(X_predict)
    
            pred_label_denorm = [(i*(max_label-min_label))+min_label for i in predictions[:,0]]
            actual_label_denorm = df_main[df_main["Radius"] == R_actual][label]
    
    
            z = [i*1000000 for i in df_actual_fea['z']]
    
            from matplotlib.ticker import StrMethodFormatter
            
            diff = (actual_label_denorm - pred_label_denorm).abs()
            rel_error = diff / actual_label_denorm
            pct_error = rel_error * 100
            ape = pct_error.mean()
            ape_list.append(ape)
            
            fig = plt.figure(figsize=(7, 4))
            ax = plt.axes()
            ax.scatter(z, actual_label_denorm, s=s, facecolor='none', edgecolor='blue')
            ax.plot(z, pred_label_denorm, color = "red")
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
            plt.ylabel("Beam Waist\n(µm, along y-axis)")
            plt.legend(["Original Data", "Predicted Data"], prop={'weight': 'bold','size': 10}, loc = "upper left")
            plt.title(str(ape)+label+"\n"+NAME, fontweight = 'bold')
            plt.show()
            plt.close()
            print(" ")
            print(" ")
            print(" ")
            print("Mean Absolute Error"+label, mean_absolute_error(actual_label_denorm, pred_label_denorm))
            print(" ")
            print(" ")
            print(" ")
    
    df_mae = pd.DataFrame()
    df_mae["name"] = name
    df_mae["mae"] = mae
    df_mae["number of layers"] = layers1
    df_mae["number of nodes first layer"] = nodes
    df_mae["ape_list"] = ape_list
    df_mae["train_test_ape"] = train_test_ape
    df_mae["RMS"] = RMS
    L_no_overfit = []
    N_no_overfit = []
    ape_no_overfit = []
    
    # Given percentile
    percentile = 20

    # Calculate the cutoff value
    cutoff_value_fitting = np.percentile(df_mae['train_test_ape'], percentile)
    cutoff_value_rms = np.percentile(df_mae['RMS'], percentile)
    for index, row in df_mae.iterrows():
    # Check if the value in column A (assuming it's named 'X') is less than 5
        if row['train_test_ape'] < cutoff_value_fitting and row['RMS'] < cutoff_value_rms:

            print(row['number of layers'], row['number of nodes first layer'], row["ape_list"])
            L_no_overfit.append(row['number of layers'])
            N_no_overfit.append(row['number of nodes first layer'])
            ape_no_overfit.append(row["ape_list"])
    df_no_overfit = pd.DataFrame()
    df_no_overfit["L_no_overfit"] = L_no_overfit
    df_no_overfit["N_no_overfit"] = N_no_overfit
    df_no_overfit["ape_no_overfit"] = ape_no_overfit
    
    plt.plot(df_mae["mae"])
    plt.show()
    
    import seaborn as sns
    mat = df_mae.pivot('number of layers', 'number of nodes first layer', 'ape_list')
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
    
    mat = df_mae.pivot('number of layers', 'number of nodes first layer', 'RMS')
    fig, ax = plt.subplots()
    ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold", fontsize = 12)
    ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold", fontsize = 12)
    sns.heatmap(mat, annot=True, cmap='hot', annot_kws={"fontsize":10, "fontweight":"bold"}, fmt=".4f")
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight("bold")
    cbar.ax.set_title("MSE", fontweight="bold")
    font = {'color': 'black', 'weight': 'bold', 'size': 12}
    ax.set_ylabel("Number of Layers, L", fontdict=font)
    ax.set_xlabel("Number of Nodes, N\n(First Layer)", fontdict=font)
    ax.tick_params(axis='both', labelsize=12, weight='bold')

"""


        
df_mae = pd.DataFrame()
df_mae["name"] = name
df_mae["mae"] = mae
df_mae["number of layers"] = layers1
df_mae["number of nodes first layer"] = nodes
df_mae["ape_actual_ver"] = ape_actual_ver
df_mae["ape_actual_hor"] = ape_actual_hor
df_mae["ape_actual_max_field_list"] = ape_actual_max_field_list

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

mat = df_mae.pivot('number of layers', 'number of nodes first layer', 'ape_actual_max_field_list')
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
"""
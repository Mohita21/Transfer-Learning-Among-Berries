import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import random
import math
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, Conv1D, Bidirectional, GRU, Flatten, Activation, BatchNormalization,Input
from tensorflow.keras.callbacks import ModelCheckpoint

import keras
from keras.models import load_model
from keras_self_attention import SeqSelfAttention
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LSTM, Activation, Bidirectional, Flatten, TimeDistributed, SimpleRNN, Dropout, GRU, Input, Add, Multiply,Lambda
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras import optimizers, initializers
from keras.regularizers import l2
from keras.initializers import TruncatedNormal, Constant, RandomNormal
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras import backend as K 

print(tf.__version__)
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
tf.random.set_seed(seed=42)
np.random.RandomState(42)
random.seed(42)
K.clear_session()
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing

r=pd.read_excel("Raspberries.xlsx")
r=r.iloc[293:3244,-1:].values
pys=pd.read_excel("soil yield price.xlsx")
s=pys.iloc[:,2:3].values
x=pys.iloc[:,21:]

#set random seed for reproducibility
print(tf.__version__)
import os
import random
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
tf.random.set_random_seed(seed=42)
np.random.RandomState(42)
random.seed(42)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
 
data=x.values
print(data.shape)
data = series_to_supervised(data, 139)
print(data)

scaler = preprocessing.StandardScaler().fit(data)
scaler

d= scaler.transform(data)

#function to find the value of d for POV=95%
def Proportion_of_Variance(x):
    val,vec=np.linalg.eig(np.cov(x.T))
    indexes=np.argsort(np.abs(val))[::-1]
    val_sorted=val[indexes]
    val_sum=val_sorted.sum()
    for k in range(784):
        k_val_sum=val_sorted[:k+1].sum()
        POV=k_val_sum/val_sum
        if POV >= 0.95:
            break
    return(k+1)
#Function to reconstruct the compressed data

k=Proportion_of_Variance(d)
print('\n The value of d using POV = 95% is ',k)

from sklearn.decomposition import PCA

my_model = PCA(n_components=20)
d=my_model.fit_transform(d)

#Training on Raspberry

X=d
Y=r
Y=Y[:2812]

print(X.shape)
print(Y.shape)
    
x_train = X[0:int(0.8*X.shape[0])]
x_test = X[int(0.8*X.shape[0]):]
y_train = Y[0:int(0.8*Y.shape[0])]
y_test = Y[int(0.8*Y.shape[0]):]
print(x_test.shape)
print(y_test.shape)
#np.random.shuffle(x_train)
print(x_train.shape)
print(y_train.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1)

keras.backend.clear_session()

os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
tf.random.set_random_seed(seed=42)
np.random.RandomState(42)
random.seed(42)

model_lime_cnn_lstm_att = keras.models.Sequential([
      keras.layers.Conv1D(filters=120, kernel_size=3,
                          strides=1, padding="causal",
                          activation="relu",
                          input_shape=(20, 1)),
      #keras.layers.BatchNormalization(),
      keras.layers.Conv1D(filters=120, kernel_size=3,
                          strides=1, padding="causal",
                          activation="relu"),
      #keras.layers.BatchNormalization(),
      keras.layers.Conv1D(filters=120, kernel_size=3,
                          strides=1, padding="causal",
                          activation="relu"),
      #keras.layers.BatchNormalization(),
      keras.layers.Conv1D(filters=120, kernel_size=3,
                          strides=1, padding="causal",
                          activation="relu"),
      #keras.layers.BatchNormalization(),
      keras.layers.LSTM(100, return_sequences=True, activation='relu'),
      #keras.layers.Dropout(0.15),
      keras.layers.LSTM(100, return_sequences=True, activation='relu'),
      SeqSelfAttention(attention_activation='sigmoid'),
      #tf.keras.layers.Dropout(0.15),
      #tf.keras.layers.LSTM(100, return_sequences=True, activation='relu'),
      keras.layers.Flatten()])

      keras.layers.Dense(64, activation="relu"),
      #keras.layers.Dropout(0.15),
      keras.layers.Dense(32, activation="relu"),
      #keras.layers.Dropout(0.15),
      keras.layers.Dense(16, activation="relu"),
      keras.layers.Dense(1),
      keras.layers.Lambda(lambda x: x * 400)


optimizer = keras.optimizers.Adam(lr=1e-4)
model_lime_cnn_lstm_att.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=["mae"])

#Create Model Checkpoints
filepath_lime_cnn_lstm_att = "weights_lime_cnn_lstm_att_Raspberry.best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath_lime_cnn_lstm_att,
                                 monitor = 'val_loss',
                                 verbose = 0,
                                 save_best_only = True,
                                 mode = 'min')

history_lime_cnn_lstm_att = model_lime_cnn_lstm_att.fit(x_train, y_train,  validation_data= (x_test, y_test),epochs=200, callbacks = [checkpoint])

#Load best weights
model_lime_cnn_lstm_att.load_weights(filepath_lime_cnn_lstm_att)
    
plt.figure()
plt.plot(np.arange(len(history_lime_cnn_lstm_att.history['loss'])), history_lime_cnn_lstm_att.history['loss'], color='r', label='Training loss')
plt.plot(np.arange(len(history_lime_cnn_lstm_att.history['loss'])), history_lime_cnn_lstm_att.history['val_loss'], color='b', label='Validation loss')
plt.title('Training vs Validation Loss (MAE)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
    
y_pred = model_lime_cnn_lstm_att.predict(x_test)
y_pred_1x = y_pred
    
plt.figure()
plt.plot(np.arange(len(y_test)), y_test, color='r', label='Actual yield')
plt.plot(np.arange(len(y_test)), y_pred, color='b', label='Predicted yield')
plt.title('Actual vs Predicted Yield')
plt.ylabel('Yield')
plt.legend()
plt.savefig('Lime AC-LSTM 5P week(s) ahead plot.jpg', quality=100, dpi=256, optimize=True)

plt.figure()
plt.plot(np.arange(len(y_test))[:50], y_test[:50],color='r', label='Actual yield')
plt.plot(np.arange(len(y_test))[:50], y_pred[:50], color='b', label='Predicted yield')
plt.title('Zoomed in Actual vs Predicted Yield')
plt.ylabel('Yield')
plt.legend()


r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
agg_err = ((np.sqrt(mse) + mae)/2) * (1-r2)

print('R2 Score: ', r2, ' , MAE: ', mae, ' , RMSE: ',np.sqrt(mse), ' , Agg: ', agg_err)

#Training on strawberry

Y=s
Y=Y[:2812]

print(Y.shape)

y_train = Y[0:int(0.8*Y.shape[0])]
y_test = Y[int(0.8*Y.shape[0]):]
print(x_test.shape)
print(y_test.shape)
#np.random.shuffle(x_train)
print(x_train.shape)
print(y_train.shape)

filepath_lime_cnn_lstm_att2 = "weights_lime_cnn_lstm_att_Strawberry.best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath_lime_cnn_lstm_att2,
                                 monitor = 'val_loss',
                                 verbose = 0,
                                 save_best_only = True,
                                 mode = 'min')
history_lime_cnn_lstm_att = model_lime_cnn_lstm_att.fit(x_train, y_train,  validation_data= (x_test, y_test),epochs=200, callbacks = [checkpoint])


#Load best weights
model_lime_cnn_lstm_att.load_weights(filepath_lime_cnn_lstm_att2)
    
plt.figure()
plt.plot(np.arange(len(history_lime_cnn_lstm_att.history['loss'])), history_lime_cnn_lstm_att.history['loss'], color='r', label='Training loss')
plt.plot(np.arange(len(history_lime_cnn_lstm_att.history['loss'])), history_lime_cnn_lstm_att.history['val_loss'], color='b', label='Validation loss')
plt.title('Training vs Validation Loss (MAE)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('CNN-LSTM attention Lime 5P week(s) ahead - loss curve.jpg', quality=100, dpi=256, optimize=True)
    
y_pred = model_lime_cnn_lstm_att.predict(x_test)
y_pred_1x = y_pred
    
plt.figure()
plt.plot(np.arange(len(y_test)), y_test, color='r', label='Actual yield')
plt.plot(np.arange(len(y_test)), y_pred, color='b', label='Predicted yield')
plt.title('Actual vs Predicted Yield')
plt.ylabel('Yield')
plt.legend()


plt.figure()
plt.plot(np.arange(len(y_test))[:50], y_test[:50],color='r', label='Actual yield')
plt.plot(np.arange(len(y_test))[:50], y_pred[:50], color='b', label='Predicted yield')
plt.title('Zoomed in Actual vs Predicted Yield')
plt.ylabel('Yield')
plt.legend()

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
agg_err = ((np.sqrt(mse) + mae)/2) * (1-r2)

print('R2 Score: ', r2, ' , MAE: ', mae, ' , RMSE: ',np.sqrt(mse), ' , Agg: ', agg_err)
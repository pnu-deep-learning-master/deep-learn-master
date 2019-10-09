#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from keras.models import Model, Input
from keras.layers import Average, LSTM
from keras.layers.merge import Multiply
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, AveragePooling1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, History
from keras.engine import training
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical
from tensorflow.python.framework.ops import Tensor
from typing import Tuple, List


# In[2]:


def onehot_encoding(target_32bp, target_20bp):
    # one-hot encoding ACGT_32bp

    ACGT = [['A'], ['C'], ['G'], ['T']]

    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    onehot_encoder.fit(ACGT)

    RNA_encoded_32bp = []

    RNA_encoded_20bp = []

    target_32bp = np.array(target_32bp)

    target_20bp = np.array(target_20bp)

    for i in range(len(target_32bp)):
        target_32bp_raw = np.array(list(target_32bp[i])).reshape(-1, 1)

        target_32bp_encoded = onehot_encoder.transform(target_32bp_raw)

        target_20bp_raw = np.array(list(target_20bp[i])).reshape(-1, 1)

        target_20bp_encoded = onehot_encoder.transform(target_20bp_raw)

        RNA_encoded_32bp.append(target_32bp_encoded)

        RNA_encoded_20bp.append(target_20bp_encoded)

    return RNA_encoded_32bp, RNA_encoded_20bp


# In[3]:


RNA_train = pd.read_csv("./RNA_train.csv")
RNA_train.describe(include="all")
RNA_test = pd.read_csv("./RNA_test.csv")
RNA_test.describe(include="all")


train_target_34bp = RNA_train.bp_34
train_target_20bp = RNA_train.bp_20
train_indel = RNA_train.Indel


test_target_34bp = RNA_test.bp_34
test_target_20bp = RNA_test.bp_20
test_indel = RNA_test.Indel


# In[4]:


train_target_encoded_32bp, train_target_encoded_20bp = onehot_encoding(train_target_34bp, train_target_20bp)
test_target_encoded_32bp, test_target_encoded_20bp = onehot_encoding(test_target_34bp, test_target_20bp)


# In[5]:


x_train = train_target_encoded_32bp
x_train = np.array(x_train)

y_train = train_indel
y_train = list(y_train)
y_train = np.reshape(list(y_train),(-1,1)) 

x_test = test_target_encoded_32bp
x_test = np.array(x_test)

y_test = test_indel
y_test = list(y_test)
y_test = np.reshape(list(y_test),(-1,1))


# In[ ]:





# In[6]:


seed = 7
np.random.seed(seed)

batch_size = 32
epochs = 50

kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

cnn_cvscores = []
deep_conv_cvscores = []
rnn_cvscores = []
ensemble_cvscores = []


# In[7]:


for train_index, test_index in kfold.split(x_train, y_train):
    x_train2, y_train2 = x_train[train_index], y_train[train_index]
    x_test2, y_test2 = x_train[test_index], y_train[test_index]
    
    
        
    y_test2 = np.reshape(y_test2,(-1))
    y_test2 = list(y_test2)
    
    
    model_cnn = Sequential()
    model_cnn.add(Convolution1D(80, 5, activation='relu', input_shape=(34,4)))
    model_cnn.add(AveragePooling1D(2))
    model_cnn.add(Flatten())
    model_cnn.add(Dropout(0.3))
    model_cnn.add(Dense(80, activation='relu'))
    model_cnn.add(Dropout(0.3))
    model_cnn.add(Dense(40, activation='relu'))
    model_cnn.add(Dropout(0.3))
    model_cnn.add(Dense(40, activation='relu'))
    model_cnn.add(Dropout(0.3))
    model_cnn.add(Dense(1, activation='linear'))
    
    model_cnn.compile(loss='mean_squared_error', optimizer='adam')
    
    hist = model_cnn.fit(x_train2, y_train2,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1)
    
    cnn_predict_model = model_cnn.predict(x_test2, batch_size=50, verbose=0)
    cnn_predict_model = cnn_predict_model.flatten()
    cnn_predict_model = list(cnn_predict_model)

    
    cnn_com_list = np.stack([y_test2, cnn_predict_model])
    
    cnn_df = pd.DataFrame(cnn_com_list).T
    cnn_corr = cnn_df.corr(method = 'spearman')
    cnn_corr = np.array(cnn_corr)
    cnn_corr = np.reshape(cnn_corr,(-1))
    cnn_corr = list(cnn_corr)
    
    cnn_cvscores.append(cnn_corr[1])


# In[8]:


print(cnn_cvscores)


# In[ ]:





# In[9]:


import plotly.graph_objects as go
import plotly.offline as offline

fig = go.Figure()

fig.add_trace(go.Box(x=['CNN','CNN','CNN','CNN','CNN'],y=cnn_cvscores, name = 'CNN_Fold',marker=dict(size=8)))


fig.update_layout(title_text="Prediction of CRISPR-Cpf1 guid RNA activity",
    font=dict(size=15)
)

fig.show()

offline.plot(fig, filename = './cnn_fold1.html')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





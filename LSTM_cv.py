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
y_test3 = list(y_test)
y_test = np.reshape(list(y_test),(-1,1))


# In[6]:


from keras.models import load_model
encoder = load_model('encoder_34.h5')


# In[7]:


encoded_train = encoder.predict(x_train)
encoded_test = encoder.predict(x_test)


# In[8]:


seed = 7
np.random.seed(seed)

#batch_size = 32
epochs = 50

kfold = KFold(n_splits=5, shuffle=True, random_state=seed)


rnn_cvscores = []
rnn_cvscores1 = []


# In[9]:


for train_index, test_index in kfold.split(x_train, y_train):
    x_train2, y_train2 = x_train[train_index], y_train[train_index]
    x_test2, y_test2 = x_train[test_index], y_train[test_index]
    
    
    y_test2 = np.reshape(y_test2,(-1))
    y_test2 = list(y_test2)

    
    
    model_rnn = Sequential()
    model_rnn.add(LSTM(128, input_shape=(34,4,)))
    model_rnn.add(Dropout(0,2))
    model_rnn.add(Dense(1, activation='linear'))
    
    model_rnn.compile(loss='mean_squared_error', optimizer='adam')
    
    hist = model_rnn.fit(x_train2, y_train2,
                 batch_size=128,
                 epochs=epochs,
                 verbose=1)
    
    
    rnn_predict_model = model_rnn.predict(x_test2, batch_size=50, verbose=0)
    rnn_predict_model = rnn_predict_model.flatten()
    rnn_predict_model = list(rnn_predict_model)
    
    
    rnn_com_list = np.stack([y_test2, rnn_predict_model])
    
    rnn_df = pd.DataFrame(rnn_com_list).T
    rnn_corr = rnn_df.corr(method = 'spearman')
    rnn_corr = np.array(rnn_corr)
    rnn_corr = np.reshape(rnn_corr,(-1))
    rnn_corr = list(rnn_corr)
    
    rnn_cvscores.append(rnn_corr[1])
    
  


# In[10]:


for train_index, test_index in kfold.split(encoded_train, y_train):
    x_train2, y_train2 = encoded_train[train_index], y_train[train_index]
    x_test2, y_test2 = encoded_train[test_index], y_train[test_index]
        
    
    model = Sequential()
    model.add(LSTM(128, input_shape=(34,1,)))
    model.add(Dropout(0,2))
    model.add(Dense(1, activation='linear'))
    
    model.compile(loss='mse', optimizer='adam')
    
    hist = model.fit(x_train2, y_train2,
                 batch_size=128,
                 epochs=epochs,
                 verbose=1)
    
    predict_model = model.predict(encoded_test, batch_size=50, verbose=0)
    predict_model = predict_model.flatten()
    predict_model = list(predict_model)
    
    com_list = np.stack([y_test3, predict_model])
    
    df = pd.DataFrame(com_list).T
    corr = df.corr(method = 'spearman')
    corr = np.array(corr)
    corr = np.reshape(corr,(-1))
    corr = list(corr)
    
    rnn_cvscores1.append(corr[1])


# In[11]:


print(rnn_cvscores)
print(rnn_cvscores1)
#rnn_cvscores=[0.7373054556119091, 0.7234682438672786, 0.7362291152471233, 0.7490142227138831, 0.7263388539799324]

#
#rnn_cvscores1=[0.6702315648730243, 0.6696970973714246, 0.681428216662536, 0.6656406849683947, 0.681608507782907]


# In[12]:


import plotly.graph_objects as go
import plotly.offline as offline

fig = go.Figure()

fig.add_trace(go.Box(x=['LSTM','LSTM','LSTM','LSTM','LSTM'],y=rnn_cvscores, name = 'LSTM',marker=dict(size=8)))


fig.add_trace(go.Box(x=['LSTM_AE','LSTM_AE','LSTM_AE','LSTM_AE','LSTM_AE'],y=rnn_cvscores1, name = 'LSTM_AE',marker=dict(size=8)))



fig.update_layout(title_text="Prediction of CRISPR-Cpf1 guid RNA activity",
    font=dict(size=15)
)

fig.show()

offline.plot(fig, filename = './RNN_fold.html')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





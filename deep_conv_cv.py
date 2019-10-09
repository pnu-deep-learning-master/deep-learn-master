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


# In[6]:


def ensembleModels(models, model_input):
    # collect outputs of models in a list
    yModels=[model(model_input) for model in models] 
    # averaging outputs
    yAvg=keras.layers.average(yModels) 
    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg,    name='ensemble')  
   
    return modelEns


# In[7]:


seed = 7
np.random.seed(seed)

batch_size = 32
epochs = 50

kfold = KFold(n_splits=5, shuffle=True, random_state=seed)


deep_conv_cvscores1 = []
deep_conv_cvscores2 = []
deep_conv_cvscores3 = []
deep_conv_cvscores4 = []


# In[8]:


for train_index, test_index in kfold.split(x_train, y_train):
    x_train2, y_train2 = x_train[train_index], y_train[train_index]
    x_test2, y_test2 = x_train[test_index], y_train[test_index]
    
    y_test2 = np.reshape(y_test2,(-1))
    y_test2 = list(y_test2)

    model_deep_conv1 = Sequential()
    model_deep_conv1.add(Convolution1D(80, 5, activation='relu', input_shape=(34,4)))
    model_deep_conv1.add(AveragePooling1D(2))
    model_deep_conv1.add(Convolution1D(80, 5, activation='relu'))
    model_deep_conv1.add(AveragePooling1D(2))
    model_deep_conv1.add(Flatten())
    model_deep_conv1.add(Dropout(0.3))
    model_deep_conv1.add(Dense(80, activation='relu'))
    model_deep_conv1.add(Dropout(0.3))
    model_deep_conv1.add(Dense(40, activation='relu'))
    model_deep_conv1.add(Dropout(0.3))
    model_deep_conv1.add(Dense(40, activation='relu'))
    model_deep_conv1.add(Dropout(0.3))
    model_deep_conv1.add(Dense(1, activation='linear'))
    
    model_deep_conv1.compile(loss='mean_squared_error', optimizer='adam')
    
    hist1 = model_deep_conv1.fit(x_train2, y_train2,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1)
    
    
    deep_conv_predict_model1 = model_deep_conv1.predict(x_test2, batch_size=50, verbose=0)
    deep_conv_predict_model1 = deep_conv_predict_model1.flatten()
    deep_conv_predict_model1 = list(deep_conv_predict_model1)
    
    
    deep_conv_com_list1 = np.stack([y_test2, deep_conv_predict_model1])
    
    deep_conv_df1 = pd.DataFrame(deep_conv_com_list1).T
    deep_conv_corr1 = deep_conv_df1.corr(method = 'spearman')
    deep_conv_corr1 = np.array(deep_conv_corr1)
    deep_conv_corr1 = np.reshape(deep_conv_corr1,(-1))
    deep_conv_corr1 = list(deep_conv_corr1)
    
    deep_conv_cvscores1.append(deep_conv_corr1[1])
    
    
    model_deep_conv2 = Sequential()
    model_deep_conv2.add(Convolution1D(80, 5, activation='relu', input_shape=(34,4)))
    model_deep_conv2.add(AveragePooling1D(2))
    model_deep_conv2.add(Convolution1D(80, 3, activation='relu'))
    model_deep_conv2.add(AveragePooling1D(2))
    model_deep_conv2.add(Flatten())
    model_deep_conv2.add(Dropout(0.3))
    model_deep_conv2.add(Dense(80, activation='relu'))
    model_deep_conv2.add(Dropout(0.3))
    model_deep_conv2.add(Dense(40, activation='relu'))
    model_deep_conv2.add(Dropout(0.3))
    model_deep_conv2.add(Dense(40, activation='relu'))
    model_deep_conv2.add(Dropout(0.3))
    model_deep_conv2.add(Dense(1, activation='linear'))
    
    model_deep_conv2.compile(loss='mean_squared_error', optimizer='adam')
    
    hist2 = model_deep_conv2.fit(x_train2, y_train2,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1)
    
    
    deep_conv_predict_model2 = model_deep_conv2.predict(x_test2, batch_size=50, verbose=0)
    deep_conv_predict_model2 = deep_conv_predict_model2.flatten()
    deep_conv_predict_model2 = list(deep_conv_predict_model2)
    
    
    deep_conv_com_list2 = np.stack([y_test2, deep_conv_predict_model2])
    
    deep_conv_df2 = pd.DataFrame(deep_conv_com_list2).T
    deep_conv_corr2 = deep_conv_df2.corr(method = 'spearman')
    deep_conv_corr2 = np.array(deep_conv_corr2)
    deep_conv_corr2 = np.reshape(deep_conv_corr2,(-1))
    deep_conv_corr2 = list(deep_conv_corr2)
    
    deep_conv_cvscores2.append(deep_conv_corr2[1])
    
    
    model_deep_conv3 = Sequential()
    model_deep_conv3.add(Convolution1D(80, 5, activation='relu', input_shape=(34,4)))
    model_deep_conv3.add(AveragePooling1D(2))
    model_deep_conv3.add(Convolution1D(40, 5, activation='relu'))
    model_deep_conv3.add(AveragePooling1D(2))
    model_deep_conv3.add(Flatten())
    model_deep_conv3.add(Dropout(0.3))
    model_deep_conv3.add(Dense(80, activation='relu'))
    model_deep_conv3.add(Dropout(0.3))
    model_deep_conv3.add(Dense(40, activation='relu'))
    model_deep_conv3.add(Dropout(0.3))
    model_deep_conv3.add(Dense(40, activation='relu'))
    model_deep_conv3.add(Dropout(0.3))
    model_deep_conv3.add(Dense(1, activation='linear'))
    
    model_deep_conv3.compile(loss='mean_squared_error', optimizer='adam')
    
    hist3 = model_deep_conv3.fit(x_train2, y_train2,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1)
    
    
    deep_conv_predict_model3 = model_deep_conv3.predict(x_test2, batch_size=50, verbose=0)
    deep_conv_predict_model3 = deep_conv_predict_model3.flatten()
    deep_conv_predict_model3 = list(deep_conv_predict_model3)
    
    
    deep_conv_com_list3 = np.stack([y_test2, deep_conv_predict_model3])
    
    deep_conv_df3 = pd.DataFrame(deep_conv_com_list3).T
    deep_conv_corr3 = deep_conv_df3.corr(method = 'spearman')
    deep_conv_corr3 = np.array(deep_conv_corr3)
    deep_conv_corr3 = np.reshape(deep_conv_corr3,(-1))
    deep_conv_corr3 = list(deep_conv_corr3)
    
    deep_conv_cvscores3.append(deep_conv_corr3[1])
    
    model_deep_conv4 = Sequential()
    model_deep_conv4.add(Convolution1D(80, 5, activation='relu', input_shape=(34,4)))
    model_deep_conv4.add(AveragePooling1D(2))
    model_deep_conv4.add(Convolution1D(40, 3, activation='relu'))
    model_deep_conv4.add(AveragePooling1D(2))
    model_deep_conv4.add(Flatten())
    model_deep_conv4.add(Dropout(0.3))
    model_deep_conv4.add(Dense(80, activation='relu'))
    model_deep_conv4.add(Dropout(0.3))
    model_deep_conv4.add(Dense(40, activation='relu'))
    model_deep_conv4.add(Dropout(0.3))
    model_deep_conv4.add(Dense(40, activation='relu'))
    model_deep_conv4.add(Dropout(0.3))
    model_deep_conv4.add(Dense(1, activation='linear'))
    
    model_deep_conv4.compile(loss='mean_squared_error', optimizer='adam')
    
    hist4 = model_deep_conv4.fit(x_train2, y_train2,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1)
    
    
    deep_conv_predict_model4 = model_deep_conv4.predict(x_test2, batch_size=50, verbose=0)
    deep_conv_predict_model4 = deep_conv_predict_model4.flatten()
    deep_conv_predict_model4 = list(deep_conv_predict_model4)
    
    
    deep_conv_com_list4 = np.stack([y_test2, deep_conv_predict_model4])
    
    deep_conv_df4 = pd.DataFrame(deep_conv_com_list4).T
    deep_conv_corr4 = deep_conv_df4.corr(method = 'spearman')
    deep_conv_corr4 = np.array(deep_conv_corr4)
    deep_conv_corr4 = np.reshape(deep_conv_corr4,(-1))
    deep_conv_corr4 = list(deep_conv_corr4)
    
    deep_conv_cvscores4.append(deep_conv_corr4[1])


# In[9]:


print(deep_conv_cvscores1)
print(deep_conv_cvscores2)
print(deep_conv_cvscores3)
print(deep_conv_cvscores4)


# In[ ]:





# In[10]:


import plotly.graph_objects as go
import plotly.offline as offline

fig = go.Figure()

fig.add_trace(go.Box(x=['deep_conv_cvscores1','deep_conv_cvscores1','deep_conv_cvscores1','deep_conv_cvscores1','deep_conv_cvscores1'],
                     y=deep_conv_cvscores1, name = 'deep_conv_cvscores_1',marker=dict(size=8)))


fig.add_trace(go.Box(x=['deep_conv_cvscores2','deep_conv_cvscores2','deep_conv_cvscores2','deep_conv_cvscores2','deep_conv_cvscores2'],
                     y=deep_conv_cvscores2, name = 'deep_conv_cvscores_2',marker=dict(size=8)))


fig.add_trace(go.Box(x=['deep_conv_cvscores3','deep_conv_cvscores3','deep_conv_cvscores3','deep_conv_cvscores3','deep_conv_cvscores3'],
                     y=deep_conv_cvscores3, name = 'deep_conv_cvscores_3',marker=dict(size=8)))

fig.add_trace(go.Box(x=['deep_conv_cvscores4','deep_conv_cvscores4','deep_conv_cvscores4','deep_conv_cvscores4','deep_conv_cvscores4'],
                     y=deep_conv_cvscores4, name = 'deep_conv_cvscores_4',marker=dict(size=8)))



fig.update_layout(title_text="Deep_conv_cvsvores",
    yaxis=dict(title='Correlation'),
    font=dict(size=15)
)

fig.show()

offline.plot(fig, filename = './deep_conv.html')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





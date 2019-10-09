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

cnn_cvscores = []
deep_conv_cvscores = []
rnn_cvscores = []
ensemble_cvscores = []


# In[8]:


for train_index, test_index in kfold.split(x_train, y_train):
    x_train2, y_train2 = x_train[train_index], y_train[train_index]
    x_test2, y_test2 = x_train[test_index], y_train[test_index]
    
    
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
    
    y_test2 = np.reshape(y_test2,(-1))
    y_test2 = list(y_test2)
    
    cnn_com_list = np.stack([y_test2, cnn_predict_model])
    
    cnn_df = pd.DataFrame(cnn_com_list).T
    cnn_corr = cnn_df.corr(method = 'spearman')
    cnn_corr = np.array(cnn_corr)
    cnn_corr = np.reshape(cnn_corr,(-1))
    cnn_corr = list(cnn_corr)
    
    cnn_cvscores.append(cnn_corr[1])

    model_deep_conv = Sequential()
    model_deep_conv.add(Convolution1D(80, 5, activation='relu', input_shape=(34,4)))
    model_deep_conv.add(AveragePooling1D(2))
    model_deep_conv.add(Convolution1D(40, 3, activation='relu'))
    model_deep_conv.add(AveragePooling1D(2))
    model_deep_conv.add(Flatten())
    model_deep_conv.add(Dropout(0.3))
    model_deep_conv.add(Dense(80, activation='relu'))
    model_deep_conv.add(Dropout(0.3))
    model_deep_conv.add(Dense(40, activation='relu'))
    model_deep_conv.add(Dropout(0.3))
    model_deep_conv.add(Dense(40, activation='relu'))
    model_deep_conv.add(Dropout(0.3))
    model_deep_conv.add(Dense(1, activation='linear'))
    
    model_deep_conv.compile(loss='mean_squared_error', optimizer='adam')
    
    hist = model_deep_conv.fit(x_train2, y_train2,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1)
    
    
    deep_conv_predict_model = model_deep_conv.predict(x_test2, batch_size=50, verbose=0)
    deep_conv_predict_model = deep_conv_predict_model.flatten()
    deep_conv_predict_model = list(deep_conv_predict_model)
    
    
    deep_conv_com_list = np.stack([y_test2, deep_conv_predict_model])
    
    deep_conv_df = pd.DataFrame(deep_conv_com_list).T
    deep_conv_corr = deep_conv_df.corr(method = 'spearman')
    deep_conv_corr = np.array(deep_conv_corr)
    deep_conv_corr = np.reshape(deep_conv_corr,(-1))
    deep_conv_corr = list(deep_conv_corr)
    
    deep_conv_cvscores.append(deep_conv_corr[1])
    
    
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
    
    
    models = [model_cnn, model_deep_conv, model_rnn]
    
    model_input = Input(shape=models[0].input_shape[1:]) # c*h*w
    modelEns = ensembleModels(models, model_input)
    
    ensemble_model_predict = modelEns.predict(x_test2, batch_size=50, verbose=0)
    ensemble_model_predict = ensemble_model_predict.flatten()
    ensemble_model_predict = list(ensemble_model_predict)
    
    ensemble_com_list = np.stack([y_test2, ensemble_model_predict])
    
    ensemble_df = pd.DataFrame(ensemble_com_list).T
    ensemble_corr = ensemble_df.corr(method = 'spearman')
    ensemble_corr = np.array(ensemble_corr)
    ensemble_corr = np.reshape(ensemble_corr,(-1))
    ensemble_corr = list(ensemble_corr)
    
    ensemble_cvscores.append(ensemble_corr[1])


# In[9]:


print(cnn_cvscores)
print(deep_conv_cvscores)
print(rnn_cvscores)
print(ensemble_cvscores)

# all_model_nocv의 결과에서 가져옴
cnn_scores = [0.7399155672882312]
deep_conv_scores = [0.7383635099410447]
rnn_scores = [0.756011140655557]
ensemble_scores = [0.77163755704868]


# In[10]:


import plotly.graph_objects as go
import plotly.offline as offline

fig = go.Figure()

fig.add_trace(go.Box(x=['CNN','CNN','CNN','CNN','CNN'],y=cnn_cvscores, name = 'CNN_Fold',marker=dict(size=8)))
fig.add_trace(go.Scatter(x=['CNN'],y=cnn_scores,name='CNN',marker=dict(size=12)))

fig.add_trace(go.Box(x=['RNN','RNN','RNN','RNN','RNN'],y=rnn_cvscores, name = 'RNN_Fold',marker=dict(size=8)))
fig.add_trace(go.Scatter(x=['RNN'],y=rnn_scores,name='RNN',marker=dict(size=12)))

fig.add_trace(go.Box(x=['Deep','Deep','Deep','Deep','Deep'],y=deep_conv_cvscores, name = 'Deep_Fold',marker=dict(size=8)))
fig.add_trace(go.Scatter(x=['Deep'],y=deep_conv_scores,name='Deep',marker=dict(size=12)))

fig.add_trace(go.Box(x=['Ensemble','Ensemble','Ensemble','Ensemble','Ensemble'],y=ensemble_cvscores,
                     name = 'Ensemble_Fold',marker=dict(size=8)))
fig.add_trace(go.Scatter(x=['Ensemble'],y=ensemble_scores,name='Ensemble',marker=dict(size=12)))

fig.update_layout(title_text="Prediction of CRISPR-Cpf1 guid RNA activity",
    font=dict(size=15)
)

fig.show()

offline.plot(fig, filename = './model_fold.html')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





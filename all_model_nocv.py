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


batch_size = 32
epochs = 50

x_train = train_target_encoded_32bp
x_train = np.array(x_train)

y_train = train_indel
y_train = list(y_train)
y_train = np.reshape(list(y_train),(-1,1)) 

X_train, X_val, Y_train, Y_val = train_test_split (x_train,y_train,test_size=0.2,random_state=123)

X_train = np.array(X_train)
X_val = np.array(X_val)
Y_train = np.array(Y_train)
Y_val = np.array(Y_val)


x_test = test_target_encoded_32bp
x_test = np.array(x_test)

y_test = test_indel
y_test2 = list(y_test)
y_test = np.reshape(list(y_test2),(-1,1))

print(x_train.shape)


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

hist = model_cnn.fit(x_train, y_train,
             batch_size=batch_size,
             epochs=epochs,
             verbose=1)

cnn_predict_model = model_cnn.predict(x_test, batch_size=50, verbose=0)
cnn_predict_model = cnn_predict_model.flatten()
cnn_predict_model = list(cnn_predict_model)

    
cnn_com_list = np.stack([y_test2, cnn_predict_model])

cnn_df = pd.DataFrame(cnn_com_list).T
cnn_corr = cnn_df.corr(method = 'spearman')
cnn_corr = np.array(cnn_corr)
cnn_corr = np.reshape(cnn_corr,(-1))
cnn_corr = list(cnn_corr)


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

hist = model_deep_conv.fit(x_train, y_train,
             batch_size=batch_size,
             epochs=epochs,
             verbose=1)


deep_conv_predict_model = model_deep_conv.predict(x_test, batch_size=50, verbose=0)
deep_conv_predict_model = deep_conv_predict_model.flatten()
deep_conv_predict_model = list(deep_conv_predict_model)


deep_conv_com_list = np.stack([y_test2, deep_conv_predict_model])

deep_conv_df = pd.DataFrame(deep_conv_com_list).T
deep_conv_corr = deep_conv_df.corr(method = 'spearman')
deep_conv_corr = np.array(deep_conv_corr)
deep_conv_corr = np.reshape(deep_conv_corr,(-1))
deep_conv_corr = list(deep_conv_corr)


model_rnn = Sequential()
model_rnn.add(LSTM(128, input_shape=(34,4,)))
model_rnn.add(Dropout(0,2))
model_rnn.add(Dense(1, activation='linear'))

model_rnn.compile(loss='mean_squared_error', optimizer='adam')

hist = model_rnn.fit(x_train, y_train,
             batch_size=128,
             epochs=epochs,
             verbose=1)


rnn_predict_model = model_rnn.predict(x_test, batch_size=50, verbose=0)
rnn_predict_model = rnn_predict_model.flatten()
rnn_predict_model = list(rnn_predict_model)


rnn_com_list = np.stack([y_test2, rnn_predict_model])

rnn_df = pd.DataFrame(rnn_com_list).T
rnn_corr = rnn_df.corr(method = 'spearman')
rnn_corr = np.array(rnn_corr)
rnn_corr = np.reshape(rnn_corr,(-1))
rnn_corr = list(rnn_corr)

models = [model_cnn, model_deep_conv, model_rnn]

model_input = Input(shape=models[0].input_shape[1:]) # c*h*w
modelEns = ensembleModels(models, model_input)

ensemble_model_predict = modelEns.predict(x_test, batch_size=50, verbose=0)
ensemble_model_predict = ensemble_model_predict.flatten()
ensemble_model_predict = list(ensemble_model_predict)

ensemble_com_list = np.stack([y_test2, ensemble_model_predict])

ensemble_df = pd.DataFrame(ensemble_com_list).T
ensemble_corr = ensemble_df.corr(method = 'spearman')
ensemble_corr = np.array(ensemble_corr)
ensemble_corr = np.reshape(ensemble_corr,(-1))
ensemble_corr = list(ensemble_corr)


# In[8]:


print(cnn_corr[1])
print(deep_conv_corr[1])
print(rnn_corr[1])
print(ensemble_corr[1])


# In[9]:


# all_model_cv 결과물에서 가져옴
cnn_cvscores = [0.7241355700463705, 0.7328563994401415, 0.732813786792066, 0.7428520503623671, 0.7419139481600742]
deep_conv_cvscores = [0.7062785361263563, 0.7123902484417629, 0.7181328726773116, 0.7207092595890598, 0.7304804629794819]
rnn_cvscores = [0.7373054556119091, 0.7234682438672786, 0.7362291152471233, 0.7490142227138831, 0.7263388539799324]
ensemble_cvscores = [0.7435528115185887, 0.7527398920778585, 0.7475216401511622, 0.7607856092378548, 0.760021033147451]


#cnn_scores = [0.7399155672882312]
#deep_conv_scores = [0.7383635099410447]
#rnn_scores = [0.756011140655557]
#ensemble_scores = [0.77163755704868]

cnn_scores = [cnn_corr[1]]
deep_conv_scores = [deep_conv_corr[1]]
rnn_scores = [rnn_corr[1]]
ensemble_scores = [ensemble_corr[1]]



# In[10]:


import plotly.graph_objects as go
import plotly.offline as offline

fig = go.Figure()

#fig.add_trace(go.Box(x=['CNN','CNN','CNN','CNN','CNN'],y=cnn_cvscores, name = 'CNN_Fold',marker=dict(size=8)))
fig.add_trace(go.Scatter(x=['CNN'],y=cnn_scores,name='CNN',marker=dict(size=12)))

#fig.add_trace(go.Box(x=['RNN','RNN','RNN','RNN','RNN'],y=rnn_cvscores, name = 'RNN_Fold',marker=dict(size=8)))
fig.add_trace(go.Scatter(x=['RNN'],y=rnn_scores,name='RNN',marker=dict(size=12)))

#fig.add_trace(go.Box(x=['Deep','Deep','Deep','Deep','Deep'],y=deep_conv_cvscores, name = 'Deep_Fold',marker=dict(size=8)))
fig.add_trace(go.Scatter(x=['Deep'],y=deep_conv_scores,name='Deep',marker=dict(size=12)))

#fig.add_trace(go.Box(x=['Ensemble','Ensemble','Ensemble','Ensemble','Ensemble'],y=ensemble_cvscores,
                    # name = 'Ensemble_Fold',marker=dict(size=8)))
fig.add_trace(go.Scatter(x=['Ensemble'],y=ensemble_scores,name='Ensemble',marker=dict(size=12)))

fig.update_layout(title_text="Prediction of CRISPR-Cpf1 guid RNA activity",
    font=dict(size=15)
)

fig.show()

offline.plot(fig, filename = './model2-4.html')


# In[11]:


from keras.models import load_model

modelEns.save("RNA_model.h5")


# In[ ]:





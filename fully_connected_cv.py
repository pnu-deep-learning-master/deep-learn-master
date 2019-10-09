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


dense_cvscores1 = []
dense_cvscores2 = []
dense_cvscores3 = []
dense_cvscores4 = []


# In[8]:


for train_index, test_index in kfold.split(x_train, y_train):
    x_train2, y_train2 = x_train[train_index], y_train[train_index]
    x_test2, y_test2 = x_train[test_index], y_train[test_index]
    
    y_test2 = np.reshape(y_test2,(-1))
    y_test2 = list(y_test2)


    model_dense1 = Sequential()
    model_dense1.add(Dense(256, activation='relu'))
    model_dense1.add(Dropout(0.3))
    model_dense1.add(Dense(256, activation='relu'))
    model_dense1.add(Dropout(0.3))
    model_dense1.add(Dense(256, activation='relu'))
    model_dense1.add(Dropout(0.3))
    model_dense1.add(Dense(256, activation='relu'))
    model_dense1.add(Dropout(0.3))
    model_dense1.add(Flatten())
    model_dense1.add(Dense(1, activation='linear'))
    
    model_dense1.compile(loss='mean_squared_error', optimizer='adam')
    
    hist1 = model_dense1.fit(x_train2, y_train2,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1)
    
    
    dense_predict_model1 = model_dense1.predict(x_test2, batch_size=50, verbose=0)
    dense_predict_model1 = dense_predict_model1.flatten()
    dense_predict_model1 = list(dense_predict_model1)
    
    
    dense_com_list1 = np.stack([y_test2, dense_predict_model1])
    
    dense_df1 = pd.DataFrame(dense_com_list1).T
    dense_corr1 = dense_df1.corr(method = 'spearman')
    dense_corr1 = np.array(dense_corr1)
    dense_corr1 = np.reshape(dense_corr1,(-1))
    dense_corr1 = list(dense_corr1)
    
    dense_cvscores1.append(dense_corr1[1])
    
    
    model_dense2 = Sequential()
    model_dense2.add(Dense(256, activation='relu'))
    model_dense2.add(Dropout(0.3))
    model_dense2.add(Dense(256, activation='relu'))
    model_dense2.add(Dropout(0.3))
    model_dense2.add(Dense(256, activation='relu'))
    model_dense2.add(Dropout(0.3))
    model_dense2.add(Flatten())
    model_dense2.add(Dense(1, activation='linear'))
    
    model_dense2.compile(loss='mean_squared_error', optimizer='adam')
    
    hist2 = model_dense2.fit(x_train2, y_train2,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1)
    
    
    dense_predict_model2 = model_dense2.predict(x_test2, batch_size=50, verbose=0)
    dense_predict_model2 = dense_predict_model2.flatten()
    dense_predict_model2 = list(dense_predict_model2)
    
    
    dense_com_list2 = np.stack([y_test2, dense_predict_model2])
    
    dense_df2 = pd.DataFrame(dense_com_list2).T
    dense_corr2 = dense_df2.corr(method = 'spearman')
    dense_corr2 = np.array(dense_corr2)
    dense_corr2 = np.reshape(dense_corr2,(-1))
    dense_corr2 = list(dense_corr2)
    
    dense_cvscores2.append(dense_corr2[1])
    
    model_dense3 = Sequential()
    model_dense3.add(Dense(80, activation='relu'))
    model_dense3.add(Dropout(0.3))
    model_dense3.add(Dense(40, activation='relu'))
    model_dense3.add(Dropout(0.3))
    model_dense3.add(Dense(40, activation='relu'))
    model_dense3.add(Dropout(0.3))
    model_dense3.add(Flatten())
    model_dense3.add(Dense(1, activation='linear'))
    
    model_dense3.compile(loss='mean_squared_error', optimizer='adam')
    
    hist3 = model_dense3.fit(x_train2, y_train2,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1)
    
    
    dense_predict_model3 = model_dense3.predict(x_test2, batch_size=50, verbose=0)
    dense_predict_model3 = dense_predict_model3.flatten()
    dense_predict_model3 = list(dense_predict_model3)
    
    
    dense_com_list3 = np.stack([y_test2, dense_predict_model3])
    
    dense_df3 = pd.DataFrame(dense_com_list3).T
    dense_corr3 = dense_df3.corr(method = 'spearman')
    dense_corr3 = np.array(dense_corr3)
    dense_corr3 = np.reshape(dense_corr3,(-1))
    dense_corr3 = list(dense_corr3)
    
    dense_cvscores3.append(dense_corr3[1]) 

    model_dense4 = Sequential()
    model_dense4.add(Dense(80, activation='relu'))
    model_dense4.add(Dropout(0.3))
    model_dense4.add(Dense(40, activation='relu'))
    model_dense4.add(Dropout(0.3))
    model_dense4.add(Flatten())
    model_dense4.add(Dense(1, activation='linear'))
    
    model_dense4.compile(loss='mean_squared_error', optimizer='adam')
    
    hist4 = model_dense4.fit(x_train2, y_train2,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1)
    
    
    dense_predict_model4 = model_dense4.predict(x_test2, batch_size=50, verbose=0)
    dense_predict_model4 = dense_predict_model4.flatten()
    dense_predict_model4 = list(dense_predict_model4)
    
    
    dense_com_list4 = np.stack([y_test2, dense_predict_model4])
    
    dense_df4 = pd.DataFrame(dense_com_list4).T
    dense_corr4 = dense_df4.corr(method = 'spearman')
    dense_corr4 = np.array(dense_corr4)
    dense_corr4 = np.reshape(dense_corr4,(-1))
    dense_corr4 = list(dense_corr4)
    
    dense_cvscores4.append(dense_corr4[1])
    
   


# In[9]:


print(dense_cvscores1)
print(dense_cvscores2)
print(dense_cvscores3)
print(dense_cvscores4)


# In[10]:


co_plot = {
     "dense1":dense_cvscores1, 
    "dense2": dense_cvscores2, 
    "dense3": dense_cvscores3,
    "dense4": dense_cvscores4
#     "exponential({})".format(1): [np.random.exponential(1) for i in range(0, sample_size)], 

}

plt.boxplot(
    list(co_plot.values()),
    vert=True, # make the plot vertical 
    notch=False, # if it is False, it will be box
    whis=1.5 
)
plt.gca().set_xticklabels(co_plot.keys(),
                          rotation=0, fontsize=20)

plt.show()


# In[11]:


import plotly.graph_objects as go
import plotly.offline as offline

fig = go.Figure()

fig.add_trace(go.Box(x=['dense_cvscores1','dense_cvscores1','dense_cvscores1','dense_cvscores1','dense_cvscores1'],
                     y=dense_cvscores1, name = 'dense_cvscores_1',marker=dict(size=8)))


fig.add_trace(go.Box(x=['dense_cvscores2','dense_cvscores2','dense_cvscores2','dense_cvscores2','dense_cvscores2'],
                     y=dense_cvscores2, name = 'dense_cvscores_2',marker=dict(size=8)))


fig.add_trace(go.Box(x=['dense_cvscores3','dense_cvscores3','dense_cvscores3','dense_cvscores3','dense_cvscores3'],
                     y=dense_cvscores3, name = 'dense_cvscores_3',marker=dict(size=8)))

fig.add_trace(go.Box(x=['dense_cvscores4','dense_cvscores4','dense_cvscores4','dense_cvscores4','dense_cvscores4'],
                     y=dense_cvscores4, name = 'dense_cvscores_4',marker=dict(size=8)))



fig.update_layout(title_text="Dense_cvsvores",
    font=dict(size=15)
)

fig.show()

offline.plot(fig, filename = './dense.html')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





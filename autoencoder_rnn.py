#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.engine import training
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Average, LSTM, BatchNormalization
from keras.layers.convolutional import Convolution1D, AveragePooling1D, MaxPooling1D
from keras.losses import categorical_crossentropy
from keras.models import Model, Input, Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
#from tensorflow.python.framework.ops import Tensor
from typing import Tuple, List
import glob
import numpy as np
import pandas as pd
import os


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


print(y_train)


# In[6]:


batch_size = 32
epochs = 300


# In[7]:


input_img = Input(shape=(34, 4))
x = Convolution1D(32, 5, padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Convolution1D(16, 5, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Convolution1D(1, 5, padding='same')(x)
x = BatchNormalization()(x)
encoded = Activation('relu')(x)

x = Convolution1D(1, 5, padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Convolution1D(16, 5, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Convolution1D(4, 5, padding='same')(x)
x = BatchNormalization()(x)
decoded = Activation('softmax')(x)


# In[8]:


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()


# In[9]:


early_stopping = EarlyStopping(monitor='val_loss', patience=10)
history = autoencoder.fit(X_train, X_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_val, X_val),
                    callbacks=[early_stopping],
                    shuffle=True)


# In[10]:


decoded_squence = autoencoder.predict(x_test)
ary_decoded = decoded_squence
ary_test = x_test
print(decoded_squence)


# In[11]:


len = ary_decoded.shape[0]
print(len)
sequence_len = ary_decoded.shape[1]
print(sequence_len)


# In[12]:


for i in range(len):
    for j in range(sequence_len):
        maxx = np.where(max(ary_decoded[i][j]) == ary_decoded[i][j])[0][0]
        for k in range(4):
            if k==maxx :
                ary_decoded[i][j][k]=1
            else :
                ary_decoded[i][j][k]=0

ary_decoded = np.reshape(ary_decoded, (-1))
ary_test = np.reshape(x_test, (-1))
corrr = np.corrcoef(ary_decoded, ary_test)
print(corrr[0][1])


# In[13]:


cnt = 0
for i in range(1292) :
    if ary_decoded[i]!=ary_test[i] :
        cnt = cnt + 1
        
print(cnt)


# In[14]:


encoder = Model(input_img, encoded)


# In[15]:


encoded_train = encoder.predict(X_train)
encoded_val = encoder.predict(X_val)
encoded_test = encoder.predict(x_test)


# In[16]:


model = Sequential()
model.add(LSTM(128, input_shape=(34,1,)))
model.add(Dropout(0,2))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.summary()


# In[17]:


early_stopping = EarlyStopping(patience = 10)

model.fit(encoded_train, Y_train, epochs=100, batch_size=32, validation_data=(encoded_val, Y_val), callbacks=[early_stopping])


# In[18]:


predict_model = model.predict(encoded_test, batch_size=50, verbose=0)
predict_model = predict_model.flatten()

print(predict_model)


# In[19]:


list_y_test = y_test2
list_y_predict = predict_model

com_list = np.stack([list_y_test, list_y_predict])

df = pd.DataFrame(com_list).T
corr = df.corr(method = 'spearman')
print(corr)


# In[22]:


from keras.models import load_model

model.save("RNA_RNN_model.h5")
model.save_weights("RNA_RNN_model_weights.h5")


# In[ ]:





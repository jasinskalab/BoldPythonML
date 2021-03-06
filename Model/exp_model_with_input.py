from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import layers
import csv
import os
import random
import numpy as np
import pandas as pd
from scipy.io import loadmat
#from tensorflow.data import Dataset
import sys
from sklearn.model_selection import KFold






from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.keras.engine import data_adapter

import matplotlib.pyplot as plt

import sklearn.metrics


def genmodel():

    def custom_train_step(keras_model):
        original_train_step = keras_model.train_step

        def print_data_and_train_step(original_data):
            # Basically copied one-to-one from https://git.io/JvDTv
            data = data_adapter.expand_1d(original_data)
            x, y_true, w = data_adapter.unpack_x_y_sample_weight(data)
            y_pred = keras_model(x, training=True)

            # this is pretty much like on_train_batch_begin
            K.print_tensor(w, "Sample weight (w) =")
            K.print_tensor(x, "Batch input (x) =")
            K.print_tensor(y_true, "Batch output (y_true) =")
            K.print_tensor(y_pred, "Prediction (y_pred) =")

            result = original_train_step(original_data)

            # add anything here for on_train_batch_end-like behavior

            return result
        return print_data_and_train_step


    cnn=tf.keras.Sequential()
    cnn.add(layers.TimeDistributed(layers.Conv2D(96,(2,2),strides=(1,1),activation='relu'),input_shape=(672,9,5,2)))# (5,9,2,672) is the exact shape that data.mat has when loaded with loadmat. Values should be added dynamically #TODO
    cnn.add(layers.TimeDistributed(layers.MaxPool2D(pool_size=(2,2),strides=(1,1))))
    cnn.add(layers.TimeDistributed(layers.Conv2D(96,(2,2),strides=(1,1))))
    cnn.add(layers.TimeDistributed(layers.MaxPool2D(pool_size=(2,2),strides=(1,1))))
    cnn.add(layers.TimeDistributed(layers.Flatten()))
    cnn.add(layers.LSTM(units=512,input_shape=(10,512)))
    cnn.add(layers.Dense(units=64))
    cnn.add(layers.Dropout(rate=0.33))
    cnn.add(layers.Dense(units=3))
    cnn.add(layers.Softmax())
    cnn.build()
    cnn.train_step = custom_train_step(cnn)
    cnn.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])







    return cnn

if len(sys.argv)==3:
    DataPath=sys.argv[1]
    #ClassPath=sys.argv[2]
    Epochs=int(sys.argv[2])
else:
    print("Usage: model_with_input.py DataPath Epochs")
    sys.exit()




#classifierData=pd.read_excel("/home/ubuntu/BoldPythonML/Data/class.csv").to_numpy()
subjectData = loadmat(DataPath)
subjectData = subjectData['DeidentifiedData']
#classes = classifierData[:,1]
#classes=pd.read_csv(ClassPath,header=None)[0]
#reordering subject data to fit models input
#5.9.2.672->672.9.5.2


print("0")
resultarr = []
size = subjectData.shape
print(size)
label=[]
lit = 0
nl = 0
sl = 0
for x in range(0,size[1]):#0,1,2 which represent literate semiliterate nonliterate
    if subjectData[0,x]['channelData'].shape == (5,9,2,672):
        if subjectData[0,x]['score'] >= 20:
            lit = lit+1
            label.append(0)
        elif subjectData[0,x]['score'] >= 2:
            label.append(1)
            nl=nl+1
        else:
            sl=sl+1
            label.append(2)
    else:
        print(x)
for x in range(0,size[1]):
    if subjectData[0,x]['channelData'].shape == (5,9,2,672):
        print(x)
        resultarr.append(np.transpose(subjectData[0,x]['channelData'],(3,1,0,2)))
    else:
        print(x)
print("Literacy Number "+str(lit))
print("NonLit Number "+str(nl))
print("Semilit number "+str(sl))
labels = tf.keras.utils.to_categorical(label,num_classes=3)

print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

subData = np.squeeze(np.asarray(resultarr))
print(subData.shape)






'''cnn=tf.keras.Sequential()
#input shape is (n samples, image width, image height, n channels)
# 4D tensor with shape: (samples, channels, rows, cols) if data_format='channels_first' or 4D tensor with shape:
# (samples, rows, cols, channels) if data_format='channels_last'.
#https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

cnn.add(layers.TimeDistributed(layers.Conv2D(96,(2,2),strides=(1,1),activation='relu'),input_shape=(672,9,5,2)))# (5,9,2,672) is the exact shape that data.mat has when loaded with loadmat. Values should be added dynamically #TODO
cnn.add(layers.TimeDistributed(layers.MaxPool2D(pool_size=(2,2),strides=(1,1))))
cnn.add(layers.TimeDistributed(layers.Conv2D(96,(2,2),strides=(1,1))))
cnn.add(layers.TimeDistributed(layers.MaxPool2D(pool_size=(2,2),strides=(1,1))))
cnn.add(layers.TimeDistributed(layers.Flatten()))
cnn.add(layers.LSTM(units=512,input_shape=(10,512)))
cnn.add(layers.Dense(units=64))
cnn.add(layers.Dropout(rate=0.33))
cnn.add(layers.Dense(units=3))
cnn.add(layers.Softmax())
cnn.build()
print("built")'''
cnn=genmodel()
#cnn.summary()
cnn.fit(subData,labels,epochs=Epochs)


    
'''train_data,test_data=tf.split(subData,[96,24])d
train_label,test_label=tf.split(labels,[96,24])
cnn.fit(train_data,train_label,epochs=50,validation_data=(test_data,test_label))'''


#cnn.fit(dataArr,labels,epochs=1)







# x=loadmat("/Volumes/data/Data/ben_IC/MattesonWorking/Working-Code/JiamianWorking/data.mat")
# data=x['S']['channelData'][0]
# a=pd.read_csv('/Volumes/data/Data/ben_IC/MattesonWorking/Working-Code/JiamianWorking/class.csv',header=None)
# cla=a[0]
# train_data=data[:108]
# test_data=data[108:]

# real_cla=[]
# for i in cla:
#     if "Non" in i:
#         real_cla.append(1)
#     elif "Semi" in i:
#         real_cla.append(2)
#     else:
#         real_cla.append(3)
# real_cla=np.asarray(real_cla)
# train_cla=real_cla[:108]
# test_cla=real_cla[108:]
# cnn.fit(train_data,train_cla,epochs=3,validation_data=(test_data,test_cla))

"""
cnn.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
cnn=tf.keras.Sequential()
cnn.add(layers.Conv2D(kernel_size=(2,2),filters=96,input_shape=(5,9,2),activation='relu'))
cnn.add(layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))
cnn.add(layers.Conv2D(kernel_size=(2,2),filters=96))
cnn.add(layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))
cnn.add(layers.Flatten())
model=tf.keras.Sequential()
model.add(layers.TimeDistributed(cnn,input_shape=(None, 1, 5, 96)))
model.add(layers.LSTM(units=512))
model.add(layers.Dense(units=3))
model.add(layers.Dropout(rate=0.25))
model.add(layers.Dense(units=3))
model.add(layers.Softmax())
cnn.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
from scipy.io import loadmat 
x=loadmat("/Volumes/data/Data/ben_IC/MattesonWorking/Working-Code/JiamianWorking/data.mat")
data=x['S']['channelData'][0]
a=pd.read_csv('/Volumes/data/Data/ben_IC/MattesonWorking/Working-Code/JiamianWorking/class.csv',header=None)
train_data=data[:108]
test_data=data[108:]

real_cla=[]
for i in cla:
    if "Non" in i:
        real_cla.append(1)
    elif "Semi" in i:
        real_cla.append(2)
    else:
        real_cla.append(3)
real_cla=np.asarray(real_cla)
train_cla=real_cla[:108]
test_cla=real_cla[108:]
cnn.fit(train_data,train_cla,epochs=3,validation_data=(test_data,test_cla))

https://medium.com/smileinnovation/how-to-work-with-time-distributed-data-in-a-neural-network-b8b39aa4ce00



model=tf.keras.Sequential()
model.add(layers.TimeDistributed(layers.Conv2D(96,(2,2),strides=(1,1),activation='relu'),input_shape=(672,5,9,2)))
model.add(layers.TimeDistributed(layers.MaxPool2D(pool_size=(2,2),strides=(1,1))))
model.add(layers.TimeDistributed(layers.Conv2D(96,(2,2))))
model.add(layers.TimeDistributed(layers.MaxPool2D(pool_size=(2,2),strides=(1,1))))
model.add(layers.TimeDistributed(layers.Flatten()))
model.add(layers.LSTM(units=512))
model.add(layers.Dense(units=3))
model.add(layers.Dropout(rate=0.25))
model.add(layers.Dense(units=3))
model.add(layers.Softmax())
model.build()
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

"""

def generateSubjectData(folder, trialSpec, resampleRate):
    filelist=[]
    for r,d,f in os.walk(folder):
        for f1 in f:
            if ".txt" in f1.lower():
                if "/sp" in r.lower():
                    filelist.append(os.path.join(r,filelist))
    random.shuffle(filelist)
    data=pd.read_csv(txt,sep="\t",header=34)


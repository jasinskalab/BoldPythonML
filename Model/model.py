import tensorflow as tf
from tensorflow.keras import layers
import csv
import os
import random
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tensorflow.data import Dataset


classifierData=pd.read_excel("/mnt/z/Data/ben_IC/MattesonWorking/Working-Code//MachineLearning/PythonSpring2020/Data/data.xlsx").to_numpy()
subjectData = loadmat("/mnt/z/Data/ben_IC/MattesonWorking/Working-Code//MachineLearning/PythonSpring2020/Data/initialsubdata.mat")
subjectData = subjectData['data1']
classes = classifierData[:,1]
#reordering subject data to fit models input
#5.9.2.672->672.9.5.2
print("0")
resultarr = []
size = classes.shape
label=[]
for x in range(0,size[0]):
    if classes[x] == 'Literate':
        label.append(0)
    elif classes[x] == 'NonLiterate':
        label.append(1)
for x in range(0,3):
    print(x)
    resultarr.append(np.transpose(subjectData[0,x]['channelData'],(3,1,0,2)))
    

labels = tf.keras.utils.to_categorical(label,num_classes=3)

print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

dataArr = np.squeeze(np.asarray(resultarr))
print(dataArr.shape)






cnn=tf.keras.Sequential()
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
cnn.add(layers.Dropout(rate=0.25))
cnn.add(layers.Dense(units=3))
cnn.add(layers.Softmax())
cnn.build()
print("built")
cnn.compile(optimizer='Adam',loss='categorical_crossentropy')
cnn.summary()

cnn.fit(dataArr,labels,epochs=1)







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
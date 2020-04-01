import sys


import tensorflow as tf
from tensorflow.keras import layers
import csv
import os
import random
import numpy as np
import pandas as pd
from scipy.io import loadmat
#from tensorflow.data import Dataset
import tensorflow_datasets as tfdss

if len(sys.argv)==3:
    subjectPath=sys.argv[1]
    classifierPath=sys.argv[2]
else:
    print("Usage: model.py SubjectDataPath.mat ClassifierPath.csv")
    sys.exit()
    #classifierPath= "/home/ubuntu/BoldPythonML/Data/class.csv"
    #subjectPath = "/home/ubuntu/BoldPythonML/Data/FullSubjectData135.mat"
'''classifierData=pd.read_excel("E:\PythonSpring2020\Data\data.xlsx").to_numpy()#"/mnt/z/Data/ben_IC/MattesonWorking/Working-Code//MachineLearning/PythonSpring2020/Data/data.xlsx").to_numpy()
subjectData = loadmat("E:\PythonSpring2020\Data\initialsubdata.mat")#"/mnt/z/Data/ben_IC/MattesonWorking/Working-Code//MachineLearning/PythonSpring2020/Data/initialsubdata.mat")
subjectData = subjectData['data1']
classes = classifierData[:,1]
#reordering subject data to fit models input
#5.9.2.672->672.9.5.2'''
classes=pd.read_csv(classifierPath,header=None)[0]
subjectData=loadmat(subjectPath)
subjectData=subjectData['S']
print("0")
resultarr = []
size = classes.shape
label=[]
for x in range(0,size[0]):
    if subjectData[0,x]['channelData'].shape == (5,9,2,672):
        if classes[x] == 'Literate':
            label.append(0)
        elif classes[x] == 'NonLiterate':
            label.append(1)
        else:
            label.append(2)
    else:
        print(x)
for x in range(0,135):
    if subjectData[0,x]['channelData'].shape == (5,9,2,672):
        print(x)
        resultarr.append(np.transpose(subjectData[0,x]['channelData'],(3,1,0,2)))
    else:
        print(x)
    

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
cnn.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
cnn.summary()
train_data,test_data=tf.split(dataArr,[96,24])
train_label,test_label=tf.split(labels,[96,24])
cnn.fit(train_data,train_label,epochs=50,validation_data=(test_data,test_label))
cnn.evaluate(dataArr,labels,verbose=1)
#cnn.fit(dataArr,labels,epochs=10)

def generateSubjectData(folder, trialSpec, resampleRate):
    filelist=[]
    for r,d,f in os.walk(folder):
        for f1 in f:
            if ".txt" in f1.lower():
                if "/sp" in r.lower():
                    filelist.append(os.path.join(r,filelist))
    random.shuffle(filelist)
    data=pd.read_csv(txt,sep="\t",header=34)
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


import datetime
from packaging import version




from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.keras.engine import data_adapter

import matplotlib.pyplot as plt

import sklearn.metrics

class CustomLogger(tf.keras.callbacks.Callback):
  def on_train_batch_begin(self, batch, logs=None):
    print('\nTraining: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_train_batch_end(self, batch, logs=None):
    print('\nTraining: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
    


  def on_test_batch_begin(self, batch, logs=None):
    print('\nEvaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_end(self, batch, logs=None):
    print('\nEvaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))




def genmodel(outputFile):
    verbose_output = outputFile+"_verbose.txt"
    outputFile=outputFile+".txt"
    
    print(outputFile)
    def custom_train_step(keras_model):
        original_train_step = keras_model.train_step
        f = open(outputFile,'a')
        f.write("\n\n Prediction Data "+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"\n\n")
        f.close()
        def print_data_and_train_step(original_data):
            # Basically copied one-to-one from https://git.io/JvDTv
            f = open(outputFile,'a')
            
            
            data = data_adapter.expand_1d(original_data)
            x, y_true, w = data_adapter.unpack_x_y_sample_weight(data)
            y_pred = keras_model(x, training=True)
            
            # this is pretty much like on_train_batch_begin
            # K.print_tensor(w, "Sample weight (w) =")
            # tf.print(w,summarize=-1)
            # K.print_tensor(y_true, "Batch output (y_true) =")
            fileprintOut = "file://"+outputFile
            verboseFilePrint = "file://"+verbose_output
            
            # K.print_tensor(y_pred, "Prediction (y_pred) =")
            tf.print("Training argmax y_true \n", tf.math.argmax(y_true,1),summarize=-1,output_stream=fileprintOut)
            tf.print("Training argmax y_pred \n", tf.math.argmax(y_pred,1),summarize=-1,output_stream=fileprintOut)

            tf.print("Training y_true \n", y_true,summarize=-1,output_stream=verboseFilePrint)
            tf.print("Training y_pred \n", y_pred,summarize=-1,output_stream=verboseFilePrint)
            
            
            
            result = original_train_step(original_data)
            f.close()
            # add anything here for on_train_batch_end-like behavior

            return result

        return print_data_and_train_step
    def custom_test_step(keras_model):
        original_train_step = keras_model.train_step
        f = open(outputFile,'a')
        f.write("\n\n test "+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"\n\n")
        f.close()
        def print_data_and_train_step(original_data):
            # Basically copied one-to-one from https://git.io/JvDTv
            f = open(outputFile,'a')
            
            
            data = data_adapter.expand_1d(original_data)
            x, y_true, w = data_adapter.unpack_x_y_sample_weight(data)
            y_pred = keras_model(x, training=True)
            
            # this is pretty much like on_train_batch_begin
            # K.print_tensor(w, "Sample weight (w) =")
            # tf.print(w,summarize=-1)
            # K.print_tensor(y_true, "Batch output (y_true) =")
            fileprintOut = "file://"+outputFile
            verboseFilePrint = "file://"+verbose_output
            
            # K.print_tensor(y_pred, "Prediction (y_pred) =")
            tf.print("Testing argmax y_true \n", tf.math.argmax(y_true,1),summarize=-1,output_stream=fileprintOut)
            tf.print("Testing argmax y_pred \n", tf.math.argmax(y_pred,1),summarize=-1,output_stream=fileprintOut)

            tf.print("Testing y_true \n", y_true,summarize=-1,output_stream=verboseFilePrint)
            tf.print("Testing y_pred \n", y_pred,summarize=-1,output_stream=verboseFilePrint)
            
            
            
            result = original_train_step(original_data)
            f.close()
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
    cnn.test_step = custom_test_step(cnn)
    cnn.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])







    return cnn


if len(sys.argv)==3:
    DataPath=sys.argv[1]
    #ClassPath=sys.argv[2]
    Epochs=int(sys.argv[2])
else:
    print("Usage: model_with_input.py DataPath Epochs")
    sys.exit()
def initStimData():
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
    
    for x in range(0,size[1]):#0,1,2 which represent literate semiliterate nonliterate
        if subjectData[0,x]['channelData'].shape == (5,9,2,672):
            if subjectData[0,x]['score'] >= lCutoff:
                lit = lit+1
                label.append(0)
                resultarr.append(np.transpose(subjectData[0,x]['channelData'],(3,1,0,2)))
            elif subjectData[0,x]['score'] >= slCutoff:
                label.append(1)
                sl=sl+1
                resultarr.append(np.transpose(subjectData[0,x]['channelData'],(3,1,0,2)))
            else:
                nl=nl+1
                label.append(2)
                resultarr.append(np.transpose(subjectData[0,x]['channelData'],(3,1,0,2)))
        ek
    
            
    for x in range(0,size[1]):
        if subjectData[0,x]['channelData'].shape == (5,9,2,672):
            
            resultarr.append(np.transpose(subjectData[0,x]['channelData'],(3,1,0,2)))
        
            
    print("Literacy Number "+str(lit))
    print("NonLit Number "+str(nl))
    print("Semilit number "+str(sl))
    labels = tf.keras.utils.to_categorical(label,num_classes=3)
    subData = np.squeeze(np.asarray(resultarr))
    return labels,subData




def initTrinary(lCutoff,slCutoff):
    
    subjectData = loadmat(DataPath)
    subjectData = subjectData['DeidentifiedData']

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
            if subjectData[0,x]['score'] >= lCutoff:
                lit = lit+1
                label.append(0)
            elif subjectData[0,x]['score'] >= slCutoff:
                label.append(1)
                sl=sl+1
            else:
                nl=nl+1
                label.append(2)
    
            
    for x in range(0,size[1]):
        if subjectData[0,x]['channelData'].shape == (5,9,2,672):
            
            resultarr.append(np.transpose(subjectData[0,x]['channelData'],(3,1,0,2)))
        
            
    print("Literacy Number "+str(lit))
    print("NonLit Number "+str(nl))
    print("Semilit number "+str(sl))
    labels = tf.keras.utils.to_categorical(label,num_classes=3)
    subData = np.squeeze(np.asarray(resultarr))
    return labels,subData


def initBinary(lCutoff):
    




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
    
    # for x in range(0,size[1]):#0,1,2 which represent literate semiliterate nonliterate
    #     if subjectData[0,x]['channelData'].shape == (5,9,2,672):
    #         if subjectData[0,x]['score'] >= lCutoff:
    #             lit = lit+1
    #             label.append(0)
    #         elif subjectData[0,x]['score'] >= slCutoff:
    #             label.append(1)
    #             sl=sl+1
    #         else:
    #             nl=nl+1
    #             label.append(2)
    for x in range(0,size[1]):#0,1 which represent literate nonliterate
        if subjectData[0,x]['channelData'].shape == (5,9,2,672):
            if subjectData[0,x]['score'] >= lCutoff:
                lit = lit+1
                label.append(0)
            # elif subjectData[0,x]['score'] >= slCutoff:
            #     label.append(1)
            #     sl=sl+1
            else:
                nl=nl+1
                label.append(1)
            
    for x in range(0,size[1]):
        if subjectData[0,x]['channelData'].shape == (5,9,2,672):
            
            resultarr.append(np.transpose(subjectData[0,x]['channelData'],(3,1,0,2)))
        
            
    print("Literacy Number "+str(lit))
    print("NonLit Number "+str(nl))
    print("Semilit number "+str(sl))
    labels = tf.keras.utils.to_categorical(label,num_classes=3)
    subData = np.squeeze(np.asarray(resultarr))
    return labels,subData


def startModel(labels,subData,outputFile):
    logdir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    
    cnn=genmodel(outputFile)
    #cnn.summary()
    batch_size = 24
    custom_callback = CustomLogger()
    i=1
    for train_index,test_index in KFold(5).split(subData):
        

        x_train,x_test=subData[train_index],subData[test_index]
        y_train,y_test=labels[train_index],labels[test_index]

        cnn=genmodel(outputFile)
        #cnn.summary()
        cnn.fit(x_train,y_train,
            epochs=Epochs,
            batch_size=batch_size,
            use_multiprocessing=True,
            callbacks=[ custom_callback])
        
        print('Model evaluation ',
            cnn.evaluate(x_test,y_test,
            verbose=1))
        print("Fold Number: " + str(i)+"\n")
        i=i+1    

outputFile = "/home/ubuntu/BoldPythonML/Output/"+"TrinaryClassification"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


for x in range(1,10):
    for y in range(x+1, 10):
        semilitValue = x*5
        litValue = y*5
        labels, subdata = initTrinary(litValue, semilitValue)
        startModel(labels,subdata,outputFile)

def generateSubjectData(folder, trialSpec, resampleRate):
    filelist=[]
    for r,d,f in os.walk(folder):
        for f1 in f:
            if ".txt" in f1.lower():
                if "/sp" in r.lower():
                    filelist.append(os.path.join(r,filelist))
    random.shuffle(filelist)
    data=pd.read_csv(txt,sep="\t",header=34)


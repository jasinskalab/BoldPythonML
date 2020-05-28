


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




def genmodel(outputFile,numClasses):
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
    cnn.add(layers.TimeDistributed(layers.Conv2D(96,(2,2),strides=(1,1),activation='relu'),input_shape=(405,9,5,2)))# (5,9,2,672) is the exact shape that data.mat has when loaded with loadmat. Values should be added dynamically #TODO
    cnn.add(layers.TimeDistributed(layers.MaxPool2D(pool_size=(2,2),strides=(1,1))))
    cnn.add(layers.TimeDistributed(layers.Conv2D(96,(2,2),strides=(1,1))))
    cnn.add(layers.TimeDistributed(layers.MaxPool2D(pool_size=(2,2),strides=(1,1))))
    cnn.add(layers.TimeDistributed(layers.Flatten()))
    cnn.add(layers.LSTM(units=512,input_shape=(10,512)))
    cnn.add(layers.Dense(units=64))
    cnn.add(layers.Dropout(rate=0.33))
    cnn.add(layers.Dense(units=numClasses))
    cnn.add(layers.Softmax())
    cnn.build()
    cnn.train_step = custom_train_step(cnn)
    cnn.test_step = custom_test_step(cnn)
    cnn.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])







    return cnn



def genStimModel(outputFile,numClasses):
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
    cnn.add(layers.TimeDistributed(layers.Conv2D(96,(2,2),strides=(1,1),activation='relu'),input_shape=(148,9,5,2)))# (5,9,2,672) is the exact shape that data.mat has when loaded with loadmat. Values should be added dynamically #TODO
    cnn.add(layers.TimeDistributed(layers.MaxPool2D(pool_size=(2,2),strides=(1,1))))
    cnn.add(layers.TimeDistributed(layers.Conv2D(96,(2,2),strides=(1,1))))
    cnn.add(layers.TimeDistributed(layers.MaxPool2D(pool_size=(2,2),strides=(1,1))))
    cnn.add(layers.TimeDistributed(layers.Flatten()))
    cnn.add(layers.LSTM(units=512,input_shape=(10,512)))
    cnn.add(layers.Dense(units=64))
    cnn.add(layers.Dropout(rate=0.33))
    cnn.add(layers.Dense(units=numClasses))
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

def genTrainData(subjectData,uniqueName):
    trainData = []
    testData = []
    trainLabel = [] 
    testLabel = []
    for i in range(subjectData.shape[1]):
        
        if(subjectData["subjectName"][0][i]==uniqueName and subjectData['channelData'][0][i].shape == (148,2,9,5)):
            if(len(testData)==0):
                testData = np.array(([subjectData["channelData"][0][i]]))
                testLabel = np.array(([subjectData["classifier"][0][i][0]]))
            else:
                testData = np.concatenate((testData,[subjectData["channelData"][0][i]]),axis=0)
                testLabel = np.concatenate((testLabel,[subjectData["classifier"][0][i][0]]),axis=0)
        elif(subjectData['channelData'][0][i].shape == (148,2,9,5)):
            if(len(trainData)==0):
                trainData = np.array(([subjectData["channelData"][0][i]]))
                trainLabel = np.array(([subjectData["classifier"][0][i][0]]))
            else:    
                trainData = np.concatenate((trainData,[subjectData["channelData"][0][i]]),axis=0)
                trainLabel = np.concatenate((trainLabel,[subjectData["classifier"][0][i][0]]),axis=0)
        else:
            print(i)
    return trainData,testData,trainLabel,testLabel
def initAuditoryStimData(subjectData, uniqueName):
    
    #This is a gruelling way to 
    
    #This is a gruelling way to extract the unique subjectID codes from the names. This will be iterated over to find the folds

    
    

    trainData,testData,trainLabel,testLabel = genTrainData(subjectData,uniqueName)
    trainlabels = []
    testlabels = []
    for x in range(0,trainLabel.shape[0]): #0,1,2 which represent literate semiliterate nonliterate
        if(trainLabel[x]=='audpseudo'):
            trainlabels.append(0)
        elif(trainLabel[x]=='audword'):
            trainlabels.append(1)
        elif(trainLabel[x]=='audvocoded'):
            trainlabels.append(2)
    
    for x in range(0,testLabel.shape[0]): #0,1,2 which represent literate semiliterate nonliterate
        if(testLabel[x]=='audpseudo'):
            testlabels.append(0)
        elif(testLabel[x]=='audword'):
            testlabels.append(1)
        elif(testLabel[x]=='audvocoded'):
            testlabels.append(2)
    trainlabels = tf.keras.utils.to_categorical(trainlabels,num_classes=3)
    testlabels = tf.keras.utils.to_categorical(testlabels,num_classes=3)
    
    
    
    return trainData,testData,trainlabels,testlabels
def initVisualStimData(subjectData, uniqueName):
    
    #This is a gruelling way to 
    
    #This is a gruelling way to extract the unique subjectID codes from the names. This will be iterated over to find the folds

    
    

    trainData,testData,trainLabel,testLabel = genTrainData(subjectData,uniqueName)
    trainlabels = []
    testlabels = []
    for x in range(0,trainLabel.shape[0]): #0,1,2 which represent literate semiliterate nonliterate
        if(trainLabel[x]=='visword'):
            trainlabels.append(0)
        elif(trainLabel[x]=='vispseudo'):
            trainlabels.append(1)
        elif(trainLabel[x]=='visfalsefont'):
            trainlabels.append(2)
    
    for x in range(0,testLabel.shape[0]): #0,1,2 which represent literate semiliterate nonliterate
        if(testLabel[x]=='visword'):
            testlabels.append(0)
        elif(testLabel[x]=='vispseudo'):
            testlabels.append(1)
        elif(testLabel[x]=='visfalsefont'):
            testlabels.append(2)
    trainlabels = tf.keras.utils.to_categorical(trainlabels,num_classes=3)
    testlabels = tf.keras.utils.to_categorical(testlabels,num_classes=3)
    
    
    
    return trainData,testData,trainlabels,testlabels
def initTrinary(lCutoff,slCutoff):
    




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
def startModel(labels,subData,outputFile,numClasses,numFolds):
    logdir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    
    cnn=genmodel(outputFile)
    #cnn.summary()
    batch_size = 24
    custom_callback = CustomLogger()
    i=1
    for train_index,test_index in KFold(numFolds).split(subData):
        

        x_train,x_test=subData[train_index],subData[test_index]
        y_train,y_test=labels[train_index],labels[test_index]

        cnn=genmodel(outputFile,numClasses)
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
def startAudModel(outputFile,numClasses):
    logdir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    
    #cnn.summary()

    subjectData = loadmat(DataPath)
    subjectData = subjectData['auditory']

    custom_callback = CustomLogger()
    i=1
    names = subjectData["subjectName"][0]
    for i in range(0,names.shape[0]):
        name = names[i]
        names[i] = name[0]

    namesStr = names.astype(str)
    subjectNames = [s[0:6] for s in [s.replace('_', '') for s in namesStr]]
    subjectData["subjectName"]=subjectNames
    uniqueNames = np.unique(subjectNames)

    for x in range(0,uniqueNames.shape[0]):
        print("***********************" + uniqueNames[x] + "*********************** \n")
        trainData,testData,trainlabels,testlabels = initAuditoryStimData(subjectData,uniqueNames[x])
        testArr = []
        trainArr = []
        for i in range(0,trainData.shape[0]):
            trainArr.append(np.transpose(trainData[i],(0,2,3,1)))
        for i in range(0,testData.shape[0]):
            testArr.append(np.transpose(testData[i],(0,2,3,1)))
        testData = np.squeeze(np.asarray(testArr))
        trainData = np.squeeze(np.asarray(trainArr))
        cnn=genStimModel(outputFile,numClasses)
        #cnn.summary()
        cnn.fit(trainData,trainlabels,
            epochs=25,
            validationData=(testData,testlabels)
            )
        
        
        print("Fold Number: " + str(i)+"\n")
        i=i+1  
def startVisModel(outputFile,numClasses):
    logdir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    
    #cnn.summary()

    subjectData = loadmat(DataPath)
    subjectData = subjectData['visual']

    custom_callback = CustomLogger()
    i=1
    names = subjectData["subjectName"][0]
    for i in range(0,names.shape[0]):
        name = names[i]
        names[i] = name[0]

    namesStr = names.astype(str)
    subjectNames = [s[0:6] for s in [s.replace('_', '') for s in namesStr]]
    subjectData["subjectName"]=subjectNames
    uniqueNames = np.unique(subjectNames)

    for x in range(0,uniqueNames.shape[0]):
        print("***********************" + uniqueNames[x] + "*********************** \n")
        trainData,testData,trainlabels,testlabels = initVisualStimData(subjectData,uniqueNames[x])
        testArr = []
        trainArr = []
        for i in range(0,trainData.shape[0]):
            trainArr.append(np.transpose(trainData[i],(0,2,3,1)))
        for i in range(0,testData.shape[0]):
            testArr.append(np.transpose(testData[i],(0,2,3,1)))
        testData = np.squeeze(np.asarray(testArr))
        trainData = np.squeeze(np.asarray(trainArr))
        cnn=genStimModel(outputFile,numClasses)
        #cnn.summary()
        cnn.fit(trainData,trainlabels,
            epochs=25,
            validationData=(testData,testlabels)
            )
        
        
        print("Fold Number: " + str(i)+"\n")
        i=i+1  



outputFile = "/home/ubuntu/BoldPythonML/Output/"+"StimuliClassification"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


startAudModel(outputFile,3)
    #startModel(labels,subdata,outputFile)




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
cnn.add(layers.LSTM(numClasses=512,input_shape=(10,512)))
cnn.add(layers.Dense(numClasses=64))
cnn.add(layers.Dropout(rate=0.33))
cnn.add(layers.Dense(numClasses=3))
cnn.add(layers.Softmax())
cnn.build()
print("built")'''

'''train_data,test_data=tf.split(subData,[96,24])d
train_label,test_label=tf.split(labels,[96,24])
cnn.fit(train_data,train_label,epochs=50,validation_data=(test_data,test_label))'''

    
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
model.add(layers.LSTM(numClasses=512))
model.add(layers.Dense(numClasses=3))
model.add(layers.Dropout(rate=0.25))
model.add(layers.Dense(numClasses=3))
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
model.add(layers.LSTM(numClasses=512))
model.add(layers.Dense(numClasses=3))
model.add(layers.Dropout(rate=0.25))
model.add(layers.Dense(numClasses=3))
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


import tensorflow as tf
from tensorflow.keras import layers
import csv
import os
import random
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tf.data import Dataset

classifierData=pd.read_excel("/mnt/z/Data/ben_IC/MattesonWorking/Working-Code//MachineLearning/PythonSpring2020/Data/data.xlsx").to_numpy()
subjectData = loadmat("/mnt/z/Data/ben_IC/MattesonWorking/Working-Code//MachineLearning/PythonSpring2020/Data/initialsubdata.mat")
subjectData = subjectData['data1']
classes = classifierData[:,0]
#reordering subject data to fit models input
#5.9.2.672->672.9.5.2
print("0")
resultarr = []

for x in range(0,3):
    print(x)
    resultarr.append(np.transpose(subjectData[0,x]['channelData'],(3,1,0,2)))
    


print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

dataArr = np.squeeze(np.asarray(resultarr))
print(dataArr.shape)

features_dataset = Dataset.from_tensor_slices(tf.constant(dataArr))
labels_dataset = Dataset.from_tensor_slices(tf.contsant(classes))

dataset = tf.data.Dataset.zip((features_dataset,labels_dataset))


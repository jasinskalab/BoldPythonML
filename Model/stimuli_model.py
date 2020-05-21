
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

#This is a gruelling way to 
subjectData = loadmat("/home/ubuntu/BoldPythonML/Data/auditory_stimuli_data.mat")
subjectData = subjectData['auditory']
#This is a gruelling way to extract the unique subjectID codes from the names. This will be iterated over to find the folds

names = subjectData["subjectName"][0]
for i in range(0,names.shape[0]):
    name = names[i]
    names[i] = name[0]

namesStr = names.astype(str)
subjectNames = [s[0:6] for s in [s.replace('_', '') for s in namesStr]]

uniqueNames = np.unique(subjectNames)

df = pd.DataFrame(data={'names':subjectNames,'classifier':subjectData["classifier"][0],'channelData':subjectData["channelData"][0]})




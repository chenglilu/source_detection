#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:43:29 2021

@author: lilucheng
"""

#using previous method to determine the group and compare to real
#get accuancy


import pandas as pd
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from numpy import *


#-------------------------------------------------------
# Read the file  dataframe
data = pd.read_excel('data2.xlsx',header = None,skipfooter= 1,index_col=1)

#change into the data we need float
#train data determined from dataframe
fcms = np.zeros((928,1))  
for i in range(0,928):    
   fcms[i] = data.iloc[i+1,21]  




pre_label=np.zeros((928,1))

#classify based on fcms
for i in range(0,928):   
  if (fcms[i]>=0.37):
      pre_label[i]=3
  elif (fcms[i]<=0.05):
      pre_label[i]=1
  else:
     pre_label[i]=2
         
#real label
     real_label=np.zeros((928,1))
   
for i in range(0,928):    
   real_label[i] = data.iloc[i+1,24]   
   
#compare
   
difference_total=np.zeros((928,1))
difference_total=real_label-pre_label



idx=np.where(difference_total==0)  
idx0=idx[0]
Accurancy=len(idx0)/len(difference_total)
print('Accurancy=',Accurancy)   

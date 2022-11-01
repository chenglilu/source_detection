#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 09:42:18 2021

@author: lilucheng
"""


#ann with fine data we filter out several data which 0 pressusr


import pandas as pd
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from numpy import *
from sklearn.metrics import mean_squared_error 
from random import seed
from random import random

#-------------------------------------------------------
# Read the file  dataframe
data = pd.read_excel('data2_check.xlsx',header = None,skipfooter= 1,index_col=1)

#change into the data we need float
#train data determined from dataframe
Traindata = np.zeros((915,10))  
for i in range(0,915):  
 for j in range(0,10):  
   Traindata[i][j] = data.iloc[i+1,j+6]  


#change nan into 0
for i in range(0,915):  
 for j in range(0,10):  
  if (np.isnan(Traindata[i][j])):
      Traindata[i][j]= 0




  
   
#lable from dataframe
Group=np.zeros((915,1))
for i in range(0,915):   
   Group[i] = data.iloc[i+1,24]  

accuracy_total=np.zeros((100,1))
accuracy_train=np.zeros((100,1))
 
for time in range(0,100):
#-------------------------------------------------------
    X = Traindata
    newX=X
    #add random effect
    
    #SiO2
    for i in range(0,915):  
          #seed(0)
          value=(2*random()-1)*0.015+1
          newX[i,0]=X[i,0]*value    
          
          
    for i in range(0,915):        
     if (newX[i,0]<0):
         newX[i,0]=0
    
    #TiO2
    for i in range(0,915):
          value=(2*random()-1)*0.07+1
          newX[i,1]=X[i,1]*value   
          
          
    for i in range(0,915):        
     if (newX[i,1]<0):
         newX[i,1]=0
    
    
    
    
    #Al2O3
    for i in range(0,915):  
          value=(2*random()-1)*0.015+1
          newX[i,2]=X[i,2]*value        
          
          
    for i in range(0,915):        
     if (newX[i,2]<0):
         newX[i,2]=0
    
    
    #cr2o3
    for i in range(0,915):  
          value=(2*random()-1)*0.15+1
          newX[i,3]=X[i,0]*value        
          
          
    for i in range(0,915):        
     if (newX[i,3]<0):
         newX[i,3]=0
    
    
    
    
    #feo
    for i in range(0,915):  
          value=(2*random()-1)*0.03+1
          newX[i,4]=X[i,4]*value        
          
          
    for i in range(0,915):        
     if (newX[i,4]<0):
         newX[i,4]=0
    
    
    
    
    #mno
    for i in range(0,915):  
          value=(2*random()-1)*0.15+1
          newX[i,5]=X[i,5]*value        
          
          
    for i in range(0,915):        
     if (newX[i,5]<0):
         newX[i,5]=0
    
    
    
    
    
    
    
    #mgo
    for i in range(0,915): 
          value=(2*random()-1)*0.03+1
          newX[i,6]=X[i,6]*value        
          
          
    for i in range(0,915):        
     if (newX[i,6]<0):
         newX[i,6]=0
    
    
    
    
    
    
    #cao
    for i in range(0,915):  
          value=(2*random()-1)*0.015+1
          newX[i,7]=X[i,7]*value        
          
          
    for i in range(0,915):        
     if (newX[i,7]<0):
         newX[i,7]=0
    
    
    
    
    
    #na2o
    for i in range(0,915):  
          value=(2*random()-1)*0.15+1
          newX[i,8]=X[i,8]*value        
          
          
    for i in range(0,915):        
     if (newX[i,8]<0):
         newX[i,8]=0    
         
         
         
    
    #k2o
    for i in range(0,915):  
          value=(2*random()-1)*0.15+1
          newX[i,9]=X[i,9]*value        
          
          
    for i in range(0,915):        
     if (newX[i,9]<0):
         newX[i,9]=0
    
    
    
    y = Group
    
    
    
    newy=y
    
    X_train, X_test, y_train, y_test = train_test_split(newX, newy, train_size=0.8, random_state = 0)
    
    #default
    #clf = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 20), random_state=1)
    
    
    clf = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 20), random_state=0)
    #clf = MLPClassifier(activation='relu',solver='sgd', alpha=1e-5,hidden_layer_sizes=(20, 20), random_state=1)
    #clf = MLPClassifier(activation='relu',solver='adam', alpha=1e-5,hidden_layer_sizes=(20, 20), random_state=1)
    
    
    clf = clf.fit(X_train, y_train)
    
    
    y_result=clf.predict(newX) 
    y_result=y_result.reshape(-1,1) 
    difference_total=np.zeros((915,1))
    difference_total=newy-y_result
    idx=np.where(difference_total==0)  
    idx0=idx[0]
    Same_alldata=len(idx0)/len(y_result)
    accuracy_ANNtrain = clf.score(X_train, y_train)
    accuracy_total[time]=Same_alldata
    accuracy_train[time]=accuracy_ANNtrain


meanvalue=mean(accuracy_total)
print('Mean accuracy:', meanvalue)

stdvalue=std(accuracy_total)
print('std accuracy:', stdvalue)

meanvalue_train=mean(accuracy_train)
print('Mean accuracy of train:', meanvalue_train)

stdvalue_train=std(accuracy_train)
print('std accuracy:', stdvalue_train)
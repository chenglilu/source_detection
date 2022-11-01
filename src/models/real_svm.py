#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:43:27 2021

@author: lilucheng
"""

#this svm for real data

import pandas as pd
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt


#-------------------------------------------------------
# Read the file  dataframe
data = pd.read_excel('data2.xlsx',header = None,skipfooter= 1,index_col=1)

#change into the data we need float
#train data determined from dataframe
Traindata = np.zeros((928,10))  
for i in range(0,928):  
 for j in range(0,10):  
   Traindata[i][j] = data.iloc[i+1,j+6]  


#change nan into 0
for i in range(0,928):  
 for j in range(0,10):  
  if (np.isnan(Traindata[i][j])):
      Traindata[i][j]= 0
  
   
   
   

#lable from dataframe
Group=np.zeros((928,1))
for i in range(0,928):   
   Group[i] = data.iloc[i+1,24]  


#-------------------------------------------------------
X = Traindata
y = Group
#class_names={'Mafic','Transitional','Peridotite'}



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state = 0)

linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X_train, y_train)
sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(X_train, y_train)



    
linear_pred = linear.predict(X_test)
poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)
sig_pred = sig.predict(X_test)


# retrieve the accuracy and print it for all 4 kernel functions
accuracy_lin = linear.score(X_test, y_test)
accuracy_poly = poly.score(X_test, y_test)
accuracy_rbf = rbf.score(X_test, y_test)
accuracy_sig = sig.score(X_test, y_test)

print('Accuracy Linear Kernel test:', accuracy_lin)
print('Accuracy Polynomial Kernel test:', accuracy_poly)
print('Accuracy Radial Basis Kernel test:', accuracy_rbf)
print('Accuracy Sigmoid Kernel test:', accuracy_sig)

#visilize
# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
accuracy_svm_train_lin=linear.score(X_train,y_train)
accuracy_svm_train_poly=poly.score(X_train,y_train)
accuracy_svm_train_rbf=rbf.score(X_train,y_train)
accuracy_svm_train_sig=sig.score(X_train,y_train)

print('Accuracy Linear Kernel test:', accuracy_svm_train_lin)
print('Accuracy Polynomial Kernel test:', accuracy_svm_train_poly)
print('Accuracy Radial Basis Kernel test:', accuracy_svm_train_rbf)
print('Accuracy Sigmoid Kernel test:', accuracy_svm_train_sig)


accuracy_svm_all=poly.score(X,y)
print('Accuracy Polynomial Kernel all:', accuracy_svm_all)




plot_confusion_matrix(linear, X_train, y_train,normalize='true')  

plt.show()



#------------------------------------------------------------------------------------------
#read data from random data 2  which include mafic and transitional


datar = pd.read_excel('randomdata2.xlsx',header = None,skipfooter= 1,index_col=1)



data_random = np.zeros((1000,10))  
for i in range(0,1000):  
 for j in range(0,10):  
   data_random[i][j] = datar.iloc[i+1,j+1]  
   
data_random_result=poly.predict(data_random)

data_random_result=data_random_result.reshape(-1,1)



real_random=np.zeros((1000,1)) 
for i in range(0,1000):   
   real_random[i] = datar.iloc[i+1,46]  

difference_random=np.zeros((1000,1))
difference_random=data_random_result-real_random
idx=np.where(difference_random==0)  
idx0=idx[0]
Same_tran=len(idx0)/len(real_random)
print('Accuracy Neural network of random data:', Same_tran)


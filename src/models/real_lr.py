#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:37:51 2021

@author: lilucheng
"""

#logistic regression



import pandas as pd
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


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


clf = LogisticRegression(C=1e5, random_state=0)


clf = clf.fit(X_train, y_train)




accuracy_lr = clf.score(X_test, y_test)

print('Accuracy logistic regression test:', accuracy_lr)



accuracy_lr_train = clf.score(X_train, y_train)

print('Accuracy logistic regression train:', accuracy_lr_train)

#visilize
# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results


accuracy_lr_all=clf.score(X,y)
print('Accuracy logistic regression all:', accuracy_lr_all)


plot_confusion_matrix(clf, X_train, y_train,normalize='true')  

plt.show()

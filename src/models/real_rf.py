#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:32:23 2021

@author: lilucheng
"""

#randon forest


import pandas as pd
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

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


clf = RandomForestClassifier(max_depth=2, random_state=0)
clf = clf.fit(X_train, y_train)




accuracy_rf = clf.score(X_test, y_test)

print('Accuracy random forest test:', accuracy_rf)


#visilize
# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results



accuracy_rf_train=clf.score(X_train,y_train)
print('Accuracy random froest train:', accuracy_rf_train)

accuracy_rf_all=clf.score(X,y)
print('Accuracy random forest train:', accuracy_rf_all)

plot_confusion_matrix(clf, X_train, y_train,normalize='true')  

plt.show()



#------------------------------------------------------------------------------------------
#read natural case data

data2 = pd.read_excel('natural_case_noref.xlsx',header = None,skipfooter= 1,index_col=1)

#train data determined from dataframe
Naturedata = np.zeros((763,10))  
for i in range(0,763):  
 for j in range(0,10):  
   Naturedata[i][j] = data2.iloc[i+1,j+1]  
   
Naturedata_result=clf.predict(Naturedata)

Naturedata_result=Naturedata_result.reshape(-1,1)

#------
#result compare
Previousresult=np.zeros((763,1)) 
for i in range(0,763):   
   Previousresult[i] = data2.iloc[i+1,14]  

difference_total=np.zeros((763,1))
difference_total=Previousresult-Naturedata_result


idx=np.where(difference_total==0)  
idx0=idx[0]
Same_rate=len(idx0)/len(Previousresult)
print('Same number of all the data',Same_rate)   


#-----------------------------------------------------------------------------
data_emeishan = pd.read_excel('emeishan.xlsx',header = None,skipfooter= 1,index_col=1)

#train data determined from dataframe
Emeishandata = np.zeros((1075,10))  
for i in range(0,1075):  
 for j in range(0,10):  
   Emeishandata[i][j] = data_emeishan.iloc[i+1,j+1]  
   
Emeishandata_result=clf.predict(Emeishandata)

Emeishandata_result=Emeishandata_result.reshape(-1,1)



#------
#result compare
#result compare
Previousemeishan=np.zeros((1075,1)) 
for i in range(0,1075):   
   Previousemeishan[i] = data_emeishan.iloc[i+1,12]  

difference_total=np.zeros((1075,1))
difference_total=Previousemeishan-Emeishandata_result


idx_v1=np.where(difference_total==0)  
idx_v10=idx_v1[0]
Same_rate_v1=len(idx_v10)/len(Previousemeishan)
print('Same number of emeishan_version1',Same_rate_v1)  
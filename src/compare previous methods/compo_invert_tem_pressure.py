#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:25:09 2021

@author: lilucheng
"""
#training composition to determine the temperature or pressure
#assume that we already know it is mafic
#then,based on the composition, invert their temperature and pressure?



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



#melting degree
meltdegree=np.zeros((915,1))
for i in range(0,915):   
   meltdegree[i] = data.iloc[i+1,3]  

#temperature
temperature=np.zeros((915,1))
for i in range(0,915):   
   temperature[i] = data.iloc[i+1,2]  

#pressure
pressure=np.zeros((915,1))
for i in range(0,915):   
   pressure[i] = data.iloc[i+1,1]  
   
   
index1=np.where(Group==1)   
index_peridotite=index1[0]

index2=np.where(Group==2)   
index_transition=index2[0]

index3=np.where(Group==3)   
index_mafic=index3[0]


#-------------------------------------------------------
# =============================================================================
# X = Traindata
# y = Group
# 
# 
# 
# 
# newX=X
# newy=y
# 
# X_train, X_test, y_train, y_test = train_test_split(newX, newy, train_size=0.9, random_state = 0)
# 
# 
# clf = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 20), random_state=1)
# 
# 
# clf = clf.fit(X_train, y_train)
# 
# =============================================================================

#here we only consider mafic based on the data size
#mafic


meltdegree_mafic=meltdegree[index_mafic]

temperature_mafic=temperature[index_mafic]

pressure_mafic=pressure[index_mafic]


X_mafic=Traindata[index_mafic]   #traning data for mafic

md_label=np.zeros((len(meltdegree_mafic),1))
tem_label=np.zeros((len(meltdegree_mafic),1))
pre_label=np.zeros((len(meltdegree_mafic),1))


for i in range(0,len(meltdegree_mafic)):
    if meltdegree_mafic[i]<=0.2:
      md_label[i]=1
    elif meltdegree_mafic[i]>0.2 and meltdegree_mafic[i]<=0.4:
      md_label[i]=2
    elif meltdegree_mafic[i]>0.4 and meltdegree_mafic[i]<=0.6:
      md_label[i]=3
    elif meltdegree_mafic[i]>0.6 and meltdegree_mafic[i]<=0.8:
      md_label[i]=4
    else:
      md_label[i]=5

sameLabel, counts = np.unique(md_label, return_counts=True)
print('melting degree of mafic: ',100*counts/sum(counts))


      

for i in range(0,len(meltdegree_mafic)):
    if temperature_mafic[i]<=1200:
      tem_label[i]=1
    elif temperature_mafic[i]>1200 and temperature_mafic[i]<=1400:
      tem_label[i]=2
    elif temperature_mafic[i]>1400 and temperature_mafic[i]<=1600:
      tem_label[i]=3
    elif temperature_mafic[i]>1600 and temperature_mafic[i]<=1800:
      tem_label[i]=4
    else:
      tem_label[i]=5
      
sameLabel, counts = np.unique(tem_label, return_counts=True)
print('temperature of mafic: ',100*counts/sum(counts))

      

for i in range(0,len(meltdegree_mafic)):
    if pressure_mafic[i]<=2:
      pre_label[i]=1
    elif pressure_mafic[i]>2 and pressure_mafic[i]<=4:
      pre_label[i]=2
    elif pressure_mafic[i]>4 and pressure_mafic[i]<=6:
      pre_label[i]=3
    elif pressure_mafic[i]>6 and pressure_mafic[i]<=8:
      pre_label[i]=4
    else:
      pre_label[i]=5
      
sameLabel, counts = np.unique(pre_label, return_counts=True)
print('pressure of mafic: ',sameLabel)

print('pressure of mafic: ',100*counts/sum(counts))



      
# =============================================================================
# mafic

newX=X_mafic
#newy=md_label
#newy=tem_label
newy=pre_label


X_train, X_test, y_train, y_test = train_test_split(newX, newy, train_size=0.8, random_state = 0)

clf = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 30), random_state=1)

clf = clf.fit(X_train, y_train)



accuracy_ANNtrain = clf.score(X_train, y_train)

print('Accuracy Neural network of train:', accuracy_ANNtrain)


#------
  
accuracy_ANN = clf.score(X_test, y_test)    
print('Accuracy Neural network test:', accuracy_ANN)

print('----------------------------------------------------------------------')

plt.figure()



plot_confusion_matrix(clf, newX, newy,normalize='true')  

#input samples?





#-----------------------------------------------------------------------------------------------
# =============================================================================
# peridotite
#index_peridotite



meltdegree_peridotite=meltdegree[index_peridotite]

temperature_peridotite=temperature[index_peridotite]

pressure_peridotite=pressure[index_peridotite]


X_peridotite=Traindata[index_peridotite]   #traning data for mafic

md_label2=np.zeros((len(meltdegree_peridotite),1))
tem_label2=np.zeros((len(meltdegree_peridotite),1))
pre_label2=np.zeros((len(meltdegree_peridotite),1))


for i in range(0,len(meltdegree_peridotite)):
    if meltdegree_peridotite[i]<=0.2:
      md_label2[i]=1
    elif meltdegree_peridotite[i]>0.2 and meltdegree_peridotite[i]<=0.4:
      md_label2[i]=2
    elif meltdegree_peridotite[i]>0.4 and meltdegree_peridotite[i]<=0.6:
      md_label2[i]=3
    elif meltdegree_peridotite[i]>0.6 and meltdegree_peridotite[i]<=0.8:
      md_label2[i]=4
    else:
      md_label2[i]=5

sameLabel, counts = np.unique(md_label2, return_counts=True)
print('melting degree of peridotite: ',100*counts/sum(counts))
      

for i in range(0,len(meltdegree_peridotite)):
    if temperature_peridotite[i]<=1200:
      tem_label2[i]=1
    elif temperature_peridotite[i]>1200 and temperature_peridotite[i]<=1400:
      tem_label2[i]=2
    elif temperature_peridotite[i]>1400 and temperature_peridotite[i]<=1600:
      tem_label2[i]=3
    elif temperature_peridotite[i]>1600 and temperature_peridotite[i]<=1800:
      tem_label2[i]=4
    else:
      tem_label2[i]=5
      
sameLabel, counts = np.unique(tem_label2, return_counts=True)
print('temperature of peridotite: ',100*counts/sum(counts))

            

for i in range(0,len(meltdegree_peridotite)):
    if pressure_peridotite[i]<=2:
      pre_label2[i]=1
    elif pressure_peridotite[i]>2 and pressure_peridotite[i]<=4:
      pre_label2[i]=2
    elif pressure_peridotite[i]>4 and pressure_peridotite[i]<=6:
      pre_label2[i]=3
    elif pressure_peridotite[i]>6 and pressure_peridotite[i]<=8:
      pre_label2[i]=4
    else:
      pre_label2[i]=5
sameLabel, counts = np.unique(pre_label2, return_counts=True)
print('pressure of peridotite: ',100*counts/sum(counts))      
      

# =============================================================================
# mafic

newX=X_peridotite
#newy=md_label2
#newy=tem_label2
newy=pre_label2


X_train, X_test, y_train, y_test = train_test_split(newX, newy, train_size=0.7, random_state = 0)

clf = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 30), random_state=1)

clf = clf.fit(X_train, y_train)



accuracy_ANNtrain2= clf.score(X_train, y_train)

print('Accuracy Neural network of train peridotite:', accuracy_ANNtrain2)


#------
  
accuracy_ANN2 = clf.score(X_test, y_test)    
print('Accuracy Neural network test peridotite:', accuracy_ANN2)

print('----------------------------------------------------------------------')


plt.figure()



plot_confusion_matrix(clf, newX, newy,normalize='true')  


#------------------------------------------------------------------------------


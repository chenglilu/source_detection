#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:02:23 2021

@author: lilucheng
"""



import pandas as pd
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from numpy import *
from sklearn.metrics import mean_squared_error 


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



#D=X
#idq=np.where((D[:,0]<30) | (D[:,0]>65))  
#idq0=idq[0]
#newX=np.delete(D,idq0,0)

#newy=np.delete(y,idq0)
#newy=newy.reshape(-1,1)

newX=X
newy=y

ts=np.linspace(0.45,0.95,101)

test_total=[]
train_total=[]

for i in range(0,101):
    X_train, X_test, y_train, y_test = train_test_split(newX, newy, train_size=ts[i], random_state = 0)    
    clf = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 10), random_state=1) 
    clf = clf.fit(X_train, y_train)
    accuracy_test = clf.score(X_test, y_test)
    accuracy_train = clf.score(X_train, y_train)
    test_total.append(accuracy_test)
    train_total.append(accuracy_train)



#fig=plt.figure()
    
fig, ax1 = plt.subplots(constrained_layout=True)

ax1.plot(ts,train_total,'o')
ax1.plot(ts,test_total,'o')




ax1.legend(['training set', 'test set'], loc='upper center',fontsize=8)

#ax.ylim(0.8,0.94)

ax1.set_xlabel('split ratio(%)',fontsize=12)          
ax1.set_ylabel('Accuracy',fontsize=12) 


ax2 = ax1.twiny()  # type:plt.Axes


ax2.set_xticks( ax1.get_xticks() )
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels([np.round(x * 928) for x in ax1.get_xticks()])





ax2.set_xlabel("Sample size",fontsize=12)

fig.savefig('samplesize',dpi=300)


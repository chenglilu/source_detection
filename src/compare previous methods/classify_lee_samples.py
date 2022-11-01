#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 12:00:50 2021

@author: lilucheng
"""
#using machine learning to classify sample from Lee 
#first part build ML model


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

X_train, X_test, y_train, y_test = train_test_split(newX, newy, train_size=0.8, random_state = 0)


clf = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 20), random_state=1)


clf = clf.fit(X_train, y_train)




accuracy_ANN = clf.score(X_test, y_test)

print('Accuracy Neural network test:', accuracy_ANN)


#visilize
# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
accuracy_ANNtrain = clf.score(X_train, y_train)

print('Accuracy Neural network of train:', accuracy_ANNtrain)



y_result=clf.predict(newX) 
y_result=y_result.reshape(-1,1) 
difference_total=np.zeros((915,1))
difference_total=newy-y_result
idx=np.where(difference_total==0)  
idx0=idx[0]
Same_alldata=len(idx0)/len(y_result)
print('Accuracy Neural network of all data:', Same_alldata)


#sencomd input the sample from Lee et al.2009
#------------------------------------------------------------------------------------------

df = pd.read_excel('samplesofLee.xlsx',header = 0,index_col=0)
df.isnull().any()
#print(df.isnull().any())
df.fillna(value=0, inplace=True)
#print(df.isnull().any())
#print(df.columns)
df['FeOt']=df['FeO']+0.9*df['Fe2O3']


#take part of table as the input data based on the modeling requirements
#compare to previous method
df_input=df[['SiO2','TiO2','Al2O3','Cr2O3','FeOt','MnO','MgO','CaO','Na2O','K2O','T','Gpa.3','T_P_C','P_A1','T_P_A','P_A2']]

data2=df_input.iloc[:,0:10]

Num_data=len(data2)

Naturedata =data2.values   #change into array
   


Naturedata_result=clf.predict(Naturedata)

Naturedata_result=Naturedata_result.reshape(-1,1)


###chose natural_results by ANN== 1,2,3


idxd1=np.where( (Naturedata_result==1)) #
idxd10=idxd1[0]

idxd2=np.where((Naturedata_result==2))  #transitional
idxd20=idxd2[0]


idxd3=np.where((Naturedata_result==3))   #mafic
idxd30=idxd3[0]


#########



###show the results by figures
MgO=np.zeros((Num_data,1))
for i in range(0,Num_data):   
   MgO[i] = data2.iloc[i,6] 
   


CaO=np.zeros((Num_data,1))
for i in range(0,Num_data):   
  CaO[i] = data2.iloc[i,7] 
    
FeO=np.zeros((Num_data,1))
for i in range(0,Num_data):   
  FeO[i] = data2.iloc[i,4] 

SiO2=np.zeros((Num_data,1))
for i in range(0,Num_data):   
  SiO2[i] = data2.iloc[i,0]   
  

fc3ms=FeO/CaO-3*(MgO/SiO2)



###############################
plt.figure()

ax1=plt.subplot(1,1,1)
l1=plt.scatter(MgO[idxd30],CaO[idxd30],marker='^',c='',edgecolors='0.5',s=30,linewidth=1)
#plt.scatter(Mg[idxd0],fcmsd[idxd0],marker='+',c='r',edgecolors='b',s=15,linewidth=0.5)

l2=plt.scatter(MgO[idxd20],CaO[idxd20],marker='s',c='limegreen',edgecolors='k',s=30,linewidth=1)

l3=plt.scatter(MgO[idxd10],CaO[idxd10],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=30,linewidth=1)

rect = plt.Rectangle((6,4),14,12,fill='False',facecolor='none',edgecolor='k')
ax1.add_patch(rect)



plt.legend([l1,l2,l3],['Mafic','Transitional','Peridotite'], loc='upper right',fontsize=10)
plt.xlabel('MgO(wt%)',fontsize=12)

plt.ylabel('CaO(wt%)',fontsize=12)

plt.ylim(3.5,17)
plt.xlim(5,30)





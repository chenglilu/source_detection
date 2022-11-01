#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:42:58 2021

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



#------------------------------------------------------------------------------------------
#read natural case data   Hawoii volcano

data2 = pd.read_excel('natural_case_noref.xlsx',header = None,skipfooter= 1,index_col=1)



df = pd.read_excel('samplesofLee.xlsx',header = 0,index_col=0)
df.isnull().any()
#print(df.isnull().any())
df.fillna(value=0, inplace=True)
#print(df.isnull().any())
#print(df.columns)
df['FeOt']=df['FeO']+0.9*df['Fe2O3']

#take part of table as the input data based on the modeling requirements
df_input=df[['SiO2','TiO2','Al2O3','Cr2O3','FeOt','MnO','MgO','CaO','Na2O','K2O']]





#train data determined from dataframe
Naturedata = np.zeros((763,10))  
for i in range(0,763):  
 for j in range(0,10):  
   Naturedata[i][j] = data2.iloc[i+1,j+1]  
   
Naturedata_result=clf.predict(Naturedata)

Naturedata_result=Naturedata_result.reshape(-1,1)

#------



###chose natural_results by ANN== 1,2,3


idxd1=np.where( (Naturedata_result==1)) 
idxd10=idxd1[0]

idxd2=np.where((Naturedata_result==2)) 
idxd20=idxd2[0]

#idxd3=(np.where(difference_total!=0) and np.where(Naturedata_result==3)) 

idxd3=np.where((Naturedata_result==3)) 
idxd30=idxd3[0]


#########



###show the results by figures
MgO=np.zeros((763,1))
for i in range(0,763):   
   MgO[i] = data2.iloc[i+1,7] 
   
#here mg#
Mg=np.zeros((763,1))
for i in range(0,763):   
   Mg[i] = data2.iloc[i+1,16] 
   
   
cati=np.zeros((763,1))
for i in range(0,763):   
   cati[i] = data2.iloc[i+1,17] 
   
   
sica=np.zeros((763,1)) 
for i in range(0,763):   
   sica[i] = data2.iloc[i+1,18] 



fcmsd=np.zeros((763,1))
for i in range(0,763):   
   fcmsd[i] = data2.iloc[i+1,19] 


###############################
plt.figure()


plt.subplots_adjust(left=0.17, right=0.95,bottom=0.12,top=0.95, wspace=0, hspace=0)
ax1=plt.subplot(2,2,1)

xx=[0,1]
yy1=[0.37,0.37]
yy2=[0.05,0.05]
plt.plot(xx,yy1,'b--',alpha=0.3,linewidth=0.5)
plt.plot(xx,yy2,'b--',alpha=0.3,linewidth=0.5)




l1=plt.scatter(Mg[idxd30],fcmsd[idxd30],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
#plt.scatter(Mg[idxd0],fcmsd[idxd0],marker='+',c='r',edgecolors='b',s=15,linewidth=0.5)

l2=plt.scatter(Mg[idxd20],fcmsd[idxd20],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)

l3=plt.scatter(Mg[idxd10],fcmsd[idxd10],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=15,linewidth=0.5)



rect = plt.Rectangle((0.65,-1),0.2,1.55,fill='False',facecolor='none',edgecolor='k')

ax1.add_patch(rect)


#plt.legend([l1,l2],['Both_M','ANN_P'], loc='upper center',fontsize=7)

plt.figtext(0.18,0.9,'Mafic',color='black',fontsize=8)
plt.figtext(0.18,0.746,'Mafic \nand Transitional',color='black',fontsize=8)



plt.figtext(0.18,0.63,'Mafic, Transitional',color='black',fontsize=8)
plt.figtext(0.18,0.58,'and Peridotitic',color='black',fontsize=8)







plt.ylabel('FCKANTMS',fontsize=10) 
plt.xlim(0.4, 0.95)
plt.ylim(-1.05, 1.15)
plt.xticks([])


plt.figtext(0.51,0.9,'(a)',color='black',fontsize=12)
#-------------------------------
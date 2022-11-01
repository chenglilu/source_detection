#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:46:22 2021

@author: lilucheng
"""
#first determine the source then choose the confirmed one
#use the Thermo‐Barometry to determine the temperature and pressure


#step 1 : build the ANN to determine the source for natural case




import pandas as pd
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from numpy import *
from sklearn.metrics import mean_squared_error 
from numpy import *
from keras.models import Sequential
from keras.layers import Dense
import math 

import scipy.stats as stats


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

#train data determined from dataframe
Naturedata = np.zeros((763,10))  
for i in range(0,763):  
 for j in range(0,10):  
   Naturedata[i][j] = data2.iloc[i+1,j+1]  
   
Naturedata_result=clf.predict(Naturedata)

Naturedata_result=Naturedata_result.reshape(-1,1)

#------





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


###condition:   Mg<0.65 or Mg>0.85  and     fcms<-1 or fcms>0.55   and Naturedata_result==3

###chose natural_results by ANN== 1,2,3

idxd3=np.where(Naturedata_result==3)

idx_inbox_3=np.where((Mg>0.65) & (Mg<0.8) & (fcmsd>-1) & (fcmsd<0.55) & (Naturedata_result==3)) 


d=[y for y in idxd3[0] if y not in idx_inbox_3[0]]

id_outbox_30_ha=np.array(d)



data_mafic_ha=Naturedata[id_outbox_30_ha]   #confirm they are mafic 




#######input emeishan
#-----------------------------------------------------------------------------
data_emeishan = pd.read_excel('emeishan.xlsx',header = None,skipfooter= 1,index_col=1)

#change 1075 into 1240

#train data determined from dataframe
Emeishandata = np.zeros((1240,10))  
for i in range(0,1240):  
 for j in range(0,10):  
   Emeishandata[i][j] = data_emeishan.iloc[i+1,j+3]  
   
Emeishandata_result=clf.predict(Emeishandata)

Emeishandata_result=Emeishandata_result.reshape(-1,1)





###show the results by figures
MgOe=np.zeros((1240,1))
for i in range(0,1240):   
   MgOe[i] = data_emeishan.iloc[i+1,9] 
   

Na2oe=np.zeros((1240,1))
for i in range(0,1240):   
   Na2oe[i] = data_emeishan.iloc[i+1,11] 
   
   
K2oe=np.zeros((1240,1))
for i in range(0,1240):   
   K2oe[i] = data_emeishan.iloc[i+1,12] 
   
   
#version1
Fc3ms=np.zeros((1240,1))
for i in range(0,1240):   
   Fc3ms[i] = data_emeishan.iloc[i+1,15] 


#version2
Fcms_v2=np.zeros((1240,1))
for i in range(0,1240):   
   Fcms_v2[i] = data_emeishan.iloc[i+1,17] 
   
   
catie=np.zeros((1240,1))
for i in range(0,1240):   
   catie[i] = data_emeishan.iloc[i+1,13] 

sicae=np.zeros((1240,1))
for i in range(0,1240):   
   sicae[i] = data_emeishan.iloc[i+1,14]    
   
   

nak=Na2oe+K2oe

#compute Mg#

mgjh=np.zeros((1240,1))
for i in range(0,1240):   
   mgjh[i] = data_emeishan.iloc[i+1,19]    
   
###condition:   0.65<Mg#<0.85  and     -1<fcms<0.55   and Naturedata_result==3


idxd3=np.where(Naturedata_result==3)

idx_inbox_3=np.where((mgjh>0.65) & (mgjh<0.8) & (Fcms_v2>-1) & (Fcms_v2<0.55) & (Emeishandata_result==3)) 


d=[y for y in idxd3[0] if y not in idx_inbox_3[0]]

id_outbox_30_em=np.array(d)



data_mafic_em=Naturedata[id_outbox_30_em]   #confirm they are mafic for Emeishan

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#step2  build ANN2 to determine the temperature and pressure




# melting degree
meltdegree = np.zeros((915, 1))
for i in range(0, 915):
    meltdegree[i] = data.iloc[i+1, 3]

# temperature
temperature = np.zeros((915, 1))
for i in range(0, 915):
    temperature[i] = data.iloc[i+1, 2]

# pressure
pressure = np.zeros((915, 1))
for i in range(0, 915):
    pressure[i] = data.iloc[i+1, 1]


index1 = np.where(Group == 1)
index_peridotite = index1[0]

index2 = np.where(Group == 2)
index_transition = index2[0]

index3 = np.where(Group == 3)
index_mafic = index3[0]


# -------------------------------------------------------
# here we only consider mafic based on the data size
# mafic


meltdegree_mafic = meltdegree[index_mafic]

temperature_mafic = temperature[index_mafic]

pressure_mafic = pressure[index_mafic]


X_mafic = Traindata[index_mafic]  # traning data for mafic


# =============================================================================
# mafic

newX = X_mafic
# newy=md_label
# newy=tem_label
#newy=meltdegree_mafic
newy = temperature_mafic
#newy=pressure_mafic


X_train, X_test, y_train, y_test = train_test_split(
    newX, newy, train_size=0.8, random_state=0)


# =============================================================================
# model = Sequential([
#     Dense(10, activation='relu', input_shape=(10,)),
#     Dense(10, activation='relu'),
#     Dense(1, activation='sigmoid'),
# ])
#
#
# model.compile(loss='mean_squared_error', optimizer='adam')
# =============================================================================





model = Sequential()

model.add(Dense(30, input_shape=(10,)))
model.add(Dense(30, activation='relu'))


model.add(Dense(1, activation='linear'))
#model.add(Dense(1, activation='softplus'))


model.compile(optimizer='rmsprop',
              loss='mean_squared_error')

hist = model.fit(X_train, y_train,
                 batch_size=30, epochs=100,
                 validation_data=(X_test, y_test))



#----------------------------------------------loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

#--------------------------------------------accurancy

y_pred=model.predict(X_train)


y_pred=y_pred.flatten()

y_train=y_train.flatten()


r,s = stats.pearsonr(y_pred,y_train)
print('相关系数r为 = %6.3f,'% r )

rmse=math.sqrt(sum((y_pred-y_train)**2)/len(y_train))
print('RMSE= %6.1f  '  %rmse)

plt.scatter(y_train,y_pred,marker='o',facecolor='g',edgecolor='k')

name=('Corelation Coefficient: %6.2f, RMSE=%6.2f' %(r, rmse))
plt.title(name)
plt.ylabel('Predection',fontsize=12)
plt.xlabel('True',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

#plt.legend(['Train', 'Test'], loc='upper right')
#plt.xlim(1050,1750)
#plt.ylim(1050,1750)
#plt.plot([0,1],[0,1],'r-')  #melt degree
#plt.plot([1050,1750],[1050,1750],'k--',alpha=0.3)  #temperature
plt.plot([0,6],[0,6],'r-')  #pressure


plt.show()

###applied to natural case

#------------------------------------------------------------------------------------------
#using the mafic sample determined by step 1


y_ha=model.predict(data_mafic_ha)

y_em=model.predict(data_mafic_em)


plt.figure()

plt.hist(y_ha, 20, density=True, facecolor='k', alpha=0.5)
plt.hist(y_em, 20, density=True, facecolor='g', alpha=0.5)
plt.legend(['Maun Kea','Emeishan'])
plt.xlabel('Pressure',fontsize=12)
plt.ylabel('Number',fontsize=12)
plt.show()
    
#plt.savefig('mafic_twocase.png',dpi=300)

 

#plot accuracy

# -----------------------------------------------------------------------------------------------
#if sample are peridotites, we could add from here




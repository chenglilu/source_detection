#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:37:51 2021

@author: lilucheng
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:25:09 2021

@author: lilucheng
"""
# training composition to determine the temperature or pressure
# assume that we already know it is mafic
# then,based on the composition, invert their temperature and pressure?
# this version is to predict the tem and pressure based on compositions


# -------------------------------------------------------
# Read the file  dataframe
import pandas as pd
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from numpy import *
from keras.models import Sequential
from keras.layers import Dense
import math 

import scipy.stats as stats




data = pd.read_excel(
    'data2_check.xlsx', header=None, skipfooter=1, index_col=1)

# change into the data we need float
# train data determined from dataframe
Traindata = np.zeros((915, 10))
for i in range(0, 915):
    for j in range(0, 10):
        Traindata[i][j] = data.iloc[i+1, j+6]


# change nan into 0
for i in range(0, 915):
    for j in range(0, 10):
        if (np.isnan(Traindata[i][j])):
            Traindata[i][j] = 0


# lable from dataframe
Group = np.zeros((915, 1))
for i in range(0, 915):
    Group[i] = data.iloc[i+1, 24]


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
#newy = temperature_mafic
newy=pressure_mafic


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


#plt.show()
plt.savefig('mafic_corelation.png',dpi=300)

###applied to natural case

#------------------------------------------------------------------------------------------
#read natural case data   Hawoii volcano

data2 = pd.read_excel('natural_case_noref.xlsx',header = None,skipfooter= 1,index_col=1)

#train data determined from dataframe
Naturedata = np.zeros((763,10))  
for i in range(0,763):  
 for j in range(0,10):  
   Naturedata[i][j] = data2.iloc[i+1,j+1]  
   


y_ha=model.predict(Naturedata)


   
data_emeishan = pd.read_excel('emeishan.xlsx',header = None,skipfooter= 1,index_col=1)

#change 1075 into 1240

#train data determined from dataframe
Emeishandata = np.zeros((1075,10))  
for i in range(0,1075):  
 for j in range(0,10):  
   Emeishandata[i][j] = data_emeishan.iloc[i+1,j+3]  

y_em=model.predict(Emeishandata)


plt.figure()

plt.hist(y_ha, 30, density=True, facecolor='k', alpha=0.75)
plt.hist(y_em, 30, density=True, facecolor='g', alpha=0.75)
plt.legend(['Maun Kea','Emeishan'])
plt.xlabel('Pressure',fontsize=12)
plt.ylabel('Number',fontsize=12)
#plt.show()
    
plt.savefig('mafic_twocase.png',dpi=300)

 

#plot accuracy

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# =============================================================================
# peridotite
# index_peridotite


meltdegree_peridotite = meltdegree[index_peridotite]

temperature_peridotite = temperature[index_peridotite]

pressure_peridotite = pressure[index_peridotite]


X_peridotite = Traindata[index_peridotite]  # traning data for mafic


# =============================================================================
# mafic

newX = X_peridotite
# newy=md_label2
# newy=tem_label2
#newy = temperature_peridotite
newy=pressure_peridotite


X_train, X_test, y_train, y_test = train_test_split(
    newX, newy, train_size=0.7, random_state=0)

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
print('RMSE= %6.1f  ' %rmse)

plt.scatter(y_train,y_pred,marker='o',facecolor='g',edgecolor='k')

name=('Corelation Coefficient: %6.2f, RMSE=%6.2f' %(r, rmse))

plt.title(name)
plt.ylabel('Predection')
plt.xlabel('True')
#plt.legend(['Train', 'Test'], loc='upper right')
#plt.xlim(1050,1750)
#plt.ylim(1050,1750)
#plt.plot([1050,2050],[1050,2050],'k--',alpha=0.3)  #temperature
plt.plot([0,20],[0,20],'r-')  #pressure

#y_ha=model.predict(Naturedata)
#y_em=model.predict(Emeishandata)
#plt.plot(y_ha,y_ha,'b*')
#plt.plot(y_em,y_em,'r+')

#plt.show()
plt.savefig('peridotites_corelation.png',dpi=300)

# ------------------------------------------------------------------------------

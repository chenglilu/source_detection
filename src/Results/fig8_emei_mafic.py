#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 16:35:45 2021

@author: lilucheng
"""
#produce emeishan data using mafic




import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import mean_squared_error 

from keras.models import Sequential
from keras.layers import Dense

import scipy.stats as stats



data = pd.read_excel(
    'data2_check_dry.xlsx', header=None, skipfooter=1, index_col=1)

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

# dry or not from dataframe
#1 is hydrous  0 is anhydrous
Hydrous = np.zeros((915, 1))
for i in range(0, 915):
    Hydrous[i] = data.iloc[i+1, 29]


index1 = np.where((Group == 1) & (Hydrous==1))  #hydrous
#index1 = np.where(Group == 1)

index_peridotite = index1[0]

index2 = np.where((Group == 2) & (Hydrous==1))
index_transition = index2[0]

#index3 = np.where((Group == 3) & (Hydrous==1))
index3 = np.where(Group == 3)
index_mafic = index3[0]


    


meltdegree_mafic = meltdegree[index_mafic]

temperature_mafic = temperature[index_mafic]

pressure_mafic = pressure[index_mafic]


X_mafic = Traindata[index_mafic]  # traning data for mafic

hydrous_mafic=Hydrous[index_mafic]


# =============================================================================
# mafic

newX = X_mafic
# newy=md_label
# newy=tem_label
newy_md=meltdegree_mafic
newy_tem = temperature_mafic
newy_pre=pressure_mafic

#newy_pt=1000*newy_pre/newy_tem

newy_pt=newy_tem/1000

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



X_train, X_test, y_train, y_test = train_test_split(
    newX, newy_pt, train_size=0.8, random_state=0)



model = Sequential()


#for p/t
#model.add(Dense(100, input_shape=(10,)))
#model.add(Dense(100, activation='softsign')) # 0.88
#model.add(Dense(100, activation='softsign')) # 0.88
#model.add(Dense(1, activation='linear'))


#for temperature
model.add(Dense(100,activation='softsign') )

#model.add(Dense(100, activation='elu')) # 0.88
model.add(Dense(100, activation='relu')) # 0.88
model.add(Dense(100, activation='relu')) # 0.88


model.add(Dense(100, activation='relu')) # 0.88

model.add(Dense(1, activation='linear'))

#model.add(Dense(100, activation='tanh')) # 0.88


#tanh,exponential,linear

#model.add(Dense(1, activation='linear'))


model.compile(optimizer='rmsprop',
              loss='mean_squared_error')



hist = model.fit(X_train, y_train,
                 batch_size=20, epochs=300,
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



y_pred=model.predict(newX)
y_pred=y_pred.flatten()

y_train=newy_pt.flatten()



r,s = stats.pearsonr(y_pred,y_train)
print('R= %6.2f  ' %r)


my=sum(y_train)/len(y_train)


r2_1=1-sum((y_pred-y_train)**2)/sum((y_train-my)**2)
print('${R_2}$= %6.2f  ' %r2_1)




rmse=math.sqrt(sum((y_pred-y_train)**2)/len(y_train))
print('RMSE= %6.2f  '  %rmse)


#-----------------------------
index_hy= np.where(hydrous_mafic==1) 
index_hy0=index_hy[0]


index_anhy= np.where(hydrous_mafic==0)

index_anhy0=index_anhy[0]



#-----calculate the r2 and rmse for hydrous sample

y_pred_hy=y_pred[index_hy0]

y_train_hy=y_train[index_hy0]

my_hy=sum(y_train_hy)/len(y_train_hy)

r2_hy=1-sum((y_pred_hy-y_train_hy)**2)/sum((y_train_hy-my_hy)**2)
print('${R_2}$ of hydrous= %6.2f  ' %r2_hy)

rmse_hy=math.sqrt(sum((y_pred_hy-y_train_hy)**2)/len(y_train_hy))
print('RMSE of hydrous= %6.2f  '  %rmse_hy)

#-----calculate the r2 and rmse for anhydrous sample

y_pred_anhy=y_pred[index_anhy0]

y_train_anhy=y_train[index_anhy0]

my_anhy=sum(y_train_anhy)/len(y_train_anhy)

r2_anhy=1-sum((y_pred_anhy-y_train_anhy)**2)/sum((y_train_anhy-my_anhy)**2)
print('${R_2}$ of anhydrous= %6.2f  ' %r2_anhy)

rmse_anhy=math.sqrt(sum((y_pred_anhy-y_train_anhy)**2)/len(y_train_anhy))
print('RMSE of anhydrous= %6.2f  '  %rmse_anhy)



r_m,s = stats.pearsonr(y_pred,y_train)
r_m_hy,s = stats.pearsonr(y_pred_hy,y_train_hy)
r_m_anhy,s = stats.pearsonr(y_pred_anhy,y_train_anhy)



#---------------------------------------------------figure1  mafic



#==========================================================================================
#==========================================================================================
#==========================================================================================
#==========================================================================================
#------------------------------------------------------------------------------------------
#pressure


newX = X_mafic
# newy=md_label
# newy=tem_label
newy_md=meltdegree_mafic
newy_tem = temperature_mafic
newy_pre=pressure_mafic

#newy_pt=1000*newy_pre/newy_tem

newy_pt=newy_pre

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



X_train, X_test, y_train, y_test = train_test_split(
    newX, newy_pt, train_size=0.8, random_state=0)



modelp = Sequential()


#for p/t
#model.add(Dense(100, input_shape=(10,)))
#model.add(Dense(100, activation='softsign')) # 0.88
#model.add(Dense(100, activation='softsign')) # 0.88
#model.add(Dense(1, activation='linear'))


#for temperature
modelp.add(Dense(100,activation='softsign') )

modelp.add(Dense(100, activation='relu')) # 0.88
modelp.add(Dense(100, activation='relu')) # 0.88
modelp.add(Dense(100, activation='relu')) # 0.88

modelp.add(Dense(1, activation='linear'))

#model.add(Dense(100, activation='tanh')) # 0.88


#tanh,exponential,linear

#model.add(Dense(1, activation='linear'))


modelp.compile(optimizer='rmsprop',
              loss='mean_squared_error')



histp = modelp.fit(X_train, y_train,
                 batch_size=20, epochs=200,
                 validation_data=(X_test, y_test))




#--------------------------------------------accurancy

plt.subplot(1,2,2)

yp_pred=modelp.predict(newX)
yp_pred=yp_pred.flatten()

yp_train=newy_pt.flatten()



rp,sp = stats.pearsonr(yp_pred,yp_train)
print('R= %6.2f  ' %rp)


myp=sum(yp_train)/len(yp_train)


rp2_1=1-sum((yp_pred-yp_train)**2)/sum((yp_train-myp)**2)
print('${R_2}$= %6.2f  ' %rp2_1)




rmsep=math.sqrt(sum((yp_pred-yp_train)**2)/len(yp_train))
print('RMSE= %6.2f  '  %rmsep)


#----------------------------------------------loss
plt.figure()

plt.plot(histp.history['loss'])
plt.plot(histp.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()





######mafic  for emeishan 
#--------------------- emeishan
#------------------------------------calculate the P and T for emeishan LIP
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------



df_emei= pd.read_excel('emei2.xlsx',header = 0,skipfooter= 0,index_col=0)


#train data determined from dataframe
N=len(df_emei)
Emeishandata = np.zeros((N,10))  
for i in range(0,N):  
 for j in range(0,10):  
   Emeishandata[i][j] = df_emei.iloc[i,j]  
   
#####model resutls
T_Emei=model.predict(Emeishandata)
T_Emei=T_Emei.flatten()


P_Emei=modelp.predict(Emeishandata)
P_Emei=P_Emei.flatten()


plt.figure()


#data from ML models

T_Emeistar=T_Emei*1000-15*P_Emei


#plt.errorbar(T_Emeistar,P_Emei, xerr=rmse*1000,yerr=rmsep, fmt='o', mfc='r',mec='k',ecolor='k',elinewidth=0.5,capthick=0,capsize=0.2)

#--------------------- emeishan
#read the other points

p_emei_lee=df_emei['P'].values
t_emei_lee=df_emei['T'].values

plt.scatter(t_emei_lee,p_emei_lee,marker='s',facecolor='g',edgecolor='k',linewidths=0.5)

#we need change xrre and yerr based on the rmse




plt.errorbar(T_Emei*1000,P_Emei, xerr=rmse*1000,yerr=rmsep, fmt='o', mfc='r',mec='k',ecolor='k',elinewidth=0.5,capthick=0,capsize=0.2)

plt.legend(['Lee et al.2009','this study (mafic)'],loc='lower left')


ax = plt.gca()

plt.xlim(1200,1750)
plt.ylim(0,6)

ax.invert_yaxis()
plt.ylabel('Pressure (GPa)')
ax.xaxis.set_ticks_position('top')  #将x轴的位置设置在顶部

#ax.set_xticklabels(row_labels, minor=False)

ax.set_xlabel('Temperature (℃)')    
ax.xaxis.set_label_position('top') 

plt.savefig('P_and_T_Emeishan_mafic.png',dpi=300)



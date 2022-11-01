#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 11:49:52 2021

@author: lilucheng
"""
#fig8 for emeishan case


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



meltdegree_peridotite = meltdegree[index_peridotite]

temperature_peridotite = temperature[index_peridotite]

pressure_peridotite = pressure[index_peridotite]


X_peridotite = Traindata[index_peridotite]  # traning data for mafic


# =============================================================================
# peridotite

newX = X_peridotite
# newy=md_label2
# newy=tem_label2
newy_tem = temperature_peridotite
newy_pre=pressure_peridotite


newy_pt=newy_tem/1000




X_train, X_test, y_train, y_test = train_test_split(
    newX, newy_pt, train_size=0.8, random_state=0)

model = Sequential()





#for temperature
model.add(Dense(100,activation='softsign') )

model.add(Dense(100, activation='elu')) # 0.88
model.add(Dense(100, activation='relu')) # 0.88
model.add(Dense(100, activation='relu')) # 0.88

model.add(Dense(100, activation='relu')) # 0.88

model.add(Dense(1, activation='linear'))




model.compile(optimizer='rmsprop',
              loss='mean_squared_error')

hist = model.fit(X_train, y_train,
                 batch_size=20, epochs=200,
                 validation_data=(X_test, y_test))




#--------------------------------------------accurancy

y_pred=model.predict(newX)
y_pred=y_pred.flatten()
y_train=newy_pt.flatten()


rmse=math.sqrt(sum((y_pred-y_train)**2)/len(y_train))


# ========================================================================================================
# ========================================================================================================
# peridotite  pressure

newy_pt=newy_pre


X_train, X_test, y_train, y_test = train_test_split(
    newX, newy_pt, train_size=0.8, random_state=0)

modelp = Sequential()





#for temperature
modelp.add(Dense(100,activation='softsign') )
modelp.add(Dense(100,activation='softsign') )

#modelp.add(Dense(100, activation='relu')) # 0.88
#modelp.add(Dense(100, activation='relu')) # 0.88

#modelp.add(Dense(100, activation='relu')) # 0.88

modelp.add(Dense(1, activation='linear'))




modelp.compile(optimizer='rmsprop',
              loss='mean_squared_error')

histp = modelp.fit(X_train, y_train,
                 batch_size=20, epochs=200,
                 validation_data=(X_test, y_test))



#--------------------------------------------accurancy


yp_pred=modelp.predict(newX)
yp_pred=yp_pred.flatten()

yp_train=newy_pt.flatten()

rmsep=math.sqrt(sum((yp_pred-yp_train)**2)/len(yp_train))


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
plt.errorbar(T_Emei*1000,P_Emei, xerr=rmse*1000,yerr=rmsep, fmt='o', mfc='r',mec='k',ecolor='k',elinewidth=0.5,capthick=0,capsize=0.2)


#--------------------- emeishan
#read the other points



p_emei_lee=df_emei['P'].values
t_emei_lee=df_emei['T'].values

plt.scatter(t_emei_lee,p_emei_lee,marker='s',facecolor='g',edgecolor='k',linewidths=0.5)

#we need change xrre and yerr based on the rmse

ax = plt.gca()

plt.xlim(1300,1750)
plt.ylim(0,6)

ax.invert_yaxis()
plt.ylabel('Pressure (GPa)')
ax.xaxis.set_ticks_position('top')  #将x轴的位置设置在顶部

#ax.set_xticklabels(row_labels, minor=False)

ax.set_xlabel('Temperature (℃)')    
ax.xaxis.set_label_position('top') 


#plt.savefig('P_and_T_Emeishan.png',dpi=300)



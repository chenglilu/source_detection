#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 10:43:43 2021

@author: lilucheng
"""
#composition to predict temperature
#based on each source

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

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.subplots_adjust(left=0.1, right=0.95,bottom=0.12,top=0.95, wspace=0.2, hspace=0.2)


plt.scatter(1000*y_train[index_hy0],1000*y_pred[index_hy0],marker='o',facecolor='g',edgecolor='k')
plt.scatter(1000*y_train[index_anhy0],1000*y_pred[index_anhy0],marker='o',facecolor='y',edgecolor='k')

plt.legend(['Volatile-free and hydrous','Carbonated'], loc='upper left')
#name=('Overall data: ${R^2}$= %6.2f,    RMSE=%6.2f' %(r2_1, rmse))

plt.title('Mafic')



plt.ylabel('Predicted T (℃)',fontsize=12)
plt.xlabel('Experimental T (℃)',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)



plt.text(1020,1600,'Volatile-free and hydrous\nT=1120-1700 ℃',fontsize=11,weight='bold')


#name_hy=('${R^2}$= %6.2f\nRMSE=%6.2f' %(r2_hy, rmse_hy))
name_hy=('R= %6.2f\nRMSE=%6.2f ℃' %(r_m_hy, 1000*rmse_hy))

plt.text(1020,1520,name_hy)


plt.text(1500,1300,'Carbonated\nT=1050-1500℃',fontsize=11,weight='bold')
#name_anhy=('${R^2}$= %6.2f\nRMSE=%6.2f' %(r2_anhy, rmse_anhy))
name_anhy=('R= %6.2f\nRMSE=%6.2f ℃' %(r_m_anhy, 1000*rmse_anhy))


plt.text(1500,1220,name_anhy)



plt.plot([1000,2000],[1000,2000],'k-')  #p/t
plt.plot([1000,2000],[950,1950],'k--')  #p/t
plt.plot([1000,2000],[1050,2050],'k--')  #p/t
#plt.text(4.5,5.5,'+0.5')
#plt.text(5.5,4.8,'-0.5')


plt.xlim(1000,1800)
plt.ylim(1000,1800)

name=('Overall data: R= %6.2f,    RMSE=%6.2f ℃' %(r_m, 1000*rmse))
plt.text(1130,1010,name)



#plt.savefig('mafic_correlation2.png',dpi=300)


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


#-----------------------------
index_hy= np.where(hydrous_mafic==1) 
index_hy0=index_hy[0]


index_anhy= np.where(hydrous_mafic==0)

index_anhy0=index_anhy[0]



#-----calculate the r2 and rmse for hydrous sample

yp_pred_hy=yp_pred[index_hy0]

yp_train_hy=yp_train[index_hy0]

myp_hy=sum(yp_train_hy)/len(yp_train_hy)

r2p_hy=1-sum((yp_pred_hy-yp_train_hy)**2)/sum((yp_train_hy-myp_hy)**2)
print('${R_2}$ of hydrous= %6.2f  ' %r2p_hy)

rmsep_hy=math.sqrt(sum((yp_pred_hy-yp_train_hy)**2)/len(yp_train_hy))
print('RMSE of hydrous= %6.2f  '  %rmsep_hy)

#-----calculate the r2 and rmse for anhydrous sample

yp_pred_anhy=yp_pred[index_anhy0]

yp_train_anhy=yp_train[index_anhy0]

myp_anhy=sum(yp_train_anhy)/len(yp_train_anhy)

r2p_anhy=1-sum((yp_pred_anhy-yp_train_anhy)**2)/sum((yp_train_anhy-myp_anhy)**2)
print('${R_2}$ of anhydrous= %6.2f  ' %r2_anhy)

rmsep_anhy=math.sqrt(sum((yp_pred_anhy-yp_train_anhy)**2)/len(yp_train_anhy))
print('RMSE of anhydrous= %6.2f  '  %rmsep_anhy)



rp_m,sp = stats.pearsonr(yp_pred,yp_train)
rp_m_hy,sp = stats.pearsonr(yp_pred_hy,yp_train_hy)
rp_m_anhy,sp = stats.pearsonr(yp_pred_anhy,yp_train_anhy)



#---------------------------------------------------figure1  mafic

plt.scatter(yp_train[index_hy0],yp_pred[index_hy0],marker='o',facecolor='g',edgecolor='k')
plt.scatter(yp_train[index_anhy0],yp_pred[index_anhy0],marker='o',facecolor='y',edgecolor='k')

plt.legend(['Volatile-free and hydrous','Carbonated'], loc='upper left')
#name=('Overall data: ${R^2}$= %6.2f,    RMSE=%6.2f' %(r2_1, rmse))
name=('Overall data: R= %6.2f,    RMSE=%6.2f GPa' %(rp_m, rmsep))
plt.text(1,0.1,name)
#plt.title(name)

plt.title('Mafic')


#plt.text(1.2,0.2,name)
plt.ylabel('Predicted P(GPa)',fontsize=12)
plt.xlabel('Experimental P (GPa)',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)



plt.text(0.2,5,'Volatile-free and hydrous\nP=0.28-5 GPa',fontsize=11,weight='bold')
#name_hy=('${R^2}$= %6.2f\nRMSE=%6.2f' %(r2_hy, rmse_hy))
namep_hy=('R= %6.2f\nRMSE=%6.2f GPa' %(rp_m_hy, rmsep_hy))

plt.text(0.2,4.2,namep_hy)


plt.text(4,2,'Carbonated\nP=1.5-5.5 GPa',fontsize=11,weight='bold')
#name_anhy=('${R^2}$= %6.2f\nRMSE=%6.2f' %(r2_anhy, rmse_anhy))
namep_anhy=('R= %6.2f\nRMSE=%6.2f GPa' %(rp_m_anhy, rmsep_anhy))


plt.text(4,1.2,namep_anhy)



plt.plot([0,7],[0,7],'k-')  #p/t
plt.plot([0,7],[-0.5,6.5],'k--')  #p/t
plt.plot([0,7],[0.5,7.5],'k--')  #p/t
#plt.text(4.5,5.5,'+0.5')
#plt.text(5.5,4.8,'-0.5')


plt.xlim(0,7)
plt.ylim(0,7)


plt.savefig('mafic_T_and_P.png',dpi=300)
#----------------------------------------------loss
plt.figure()

plt.plot(histp.history['loss'])
plt.plot(histp.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()





#--------------------------------------------------------------------------------------
#plot together

plt.figure()



#plot error bar 
plt.errorbar(y_pred*1000,yp_pred, xerr=rmse*1000,yerr=rmsep, fmt='o', mfc='b',
         mec='k',ecolor='k',elinewidth=1,capthick=1,capsize=0)
ax = plt.gca()






plt.xlim(1000,1800)
plt.ylim(0,7)

ax.invert_yaxis()
plt.ylabel('Pressure (GPa)')
ax.xaxis.set_ticks_position('top')  #将x轴的位置设置在顶部

#ax.set_xticklabels(row_labels, minor=False)

ax.set_xlabel('Temperature (℃)')    
ax.xaxis.set_label_position('top') 





######mafic  for emeishan 

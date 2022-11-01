#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 15:18:34 2021

@author: lilucheng
"""


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



#----------------------------------------------loss
plt.figure()

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



r_p,s = stats.pearsonr(y_pred,y_train)
print('相关系数r为 = %6.3f,'%r_p)


my=sum(y_train)/len(y_train)

r2_2=1-sum((y_pred-y_train)**2)/sum((y_train-my)**2)
print('${R^2}$= %6.2f  ' %r2_2)



rmse=math.sqrt(sum((y_pred-y_train)**2)/len(y_train))
print('RMSE= %6.3f  ' %rmse)





#------------------------------------------------------------------------figure
#-------------------------------temperature




plt.figure()

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.subplots_adjust(left=0.1, right=0.95,bottom=0.12,top=0.95, wspace=0.2, hspace=0.2)

l1=plt.scatter(1000*y_train,1000*y_pred,marker='o',facecolor='g',edgecolor='k')
plt.legend(['Volatile-free and hydrous'], loc='upper left')

#name=('Hydrous\n${R^2}$:%6.2f\nRMSE=%6.2f' %(r2_2, rmse))

plt.text(1020,1650,'Volatile-free and hydrous\nT=1000-1950 ℃',fontsize=11,weight='bold')
name=('R= %6.2f\nRMSE=%6.2f ℃' %(r_p, rmse*1000))
plt.text(1020,1520,name)


plt.title('Peridotitic')



plt.ylabel('Predicted T (℃)',fontsize=12)
plt.xlabel('Experimental T (℃)',fontsize=12)



plt.plot([1000,2000],[1000,2000],'k-')  #p/t
plt.plot([1000,2000],[950,1950],'k--')  #p/t
plt.plot([1000,2000],[1050,2050],'k--')  #p/t
#plt.text(4.5,5.5,'+0.5')
#plt.text(5.5,4.8,'-0.5')


plt.xlim(1000,2000)
plt.ylim(1000,2000)




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



r_p,s = stats.pearsonr(yp_pred,yp_train)
print('相关系数r为 = %6.3f,'%r_p)


myp=sum(yp_train)/len(yp_train)

r2p_2=1-sum((yp_pred-yp_train)**2)/sum((yp_train-my)**2)
print('${R^2}$= %6.2f  ' %r2_2)



rmsep=math.sqrt(sum((yp_pred-yp_train)**2)/len(yp_train))
print('RMSE= %6.3f  ' %rmsep)



#--------------------------------------------------------------------figure
#pressure



plt.subplot(1,2,2)

l1=plt.scatter(yp_train,yp_pred,marker='o',facecolor='g',edgecolor='k')
plt.legend(['Volatile-free and hydrous'], loc='upper left')


#name=('Hydrous\n${R^2}$:%6.2f\nRMSE=%6.2f' %(r2_2, rmse))

plt.text(0.3,5,'Volatile-free and hydrous\nP=0.5-7 GPa',fontsize=11,weight='bold')
name=('R= %6.2f\nRMSE=%6.2f GPa' %(r_p, rmsep))

plt.text(0.3,4.2,name)

plt.title('Peridotitic')



plt.ylabel('Predicted P (GPa)',fontsize=12)
plt.xlabel('Experimental P (GPa)',fontsize=12)




plt.plot([0,7.5],[0,7.5],'k-')  #p/t
plt.plot([0,7.5],[-0.5,7],'k--')  #p/t
plt.plot([0,7.5],[0.5,8],'k--')  #p/t
#plt.text(4.5,5.5,'+0.5')
#plt.text(5.5,4.8,'-0.5')


plt.xlim(0,7.5)
plt.ylim(0,7.5)


plt.savefig('Peridotitic_T_and_P.png',dpi=300)



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
#plot together P and T

plt.figure()



#plot error bar 
plt.errorbar(y_pred*1000,yp_pred, xerr=rmse*1000,yerr=rmsep, fmt='o', mfc='b',
         mec='k',ecolor='k',elinewidth=1,capthick=1,capsize=0)
ax = plt.gca()






plt.xlim(1000,2000)
plt.ylim(0,8)

ax.invert_yaxis()
plt.ylabel('Pressure (GPa)')
ax.xaxis.set_ticks_position('top')  #将x轴的位置设置在顶部

#ax.set_xticklabels(row_labels, minor=False)

ax.set_xlabel('Temperature (℃)')    
ax.xaxis.set_label_position('top') 




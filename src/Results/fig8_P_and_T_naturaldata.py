#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 17:25:45 2021

@author: lilucheng
"""
#fig8  classify lee natural data compare to previous methods
#


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



# ========================================================================================================
# ========================================================================================================
# peridotite  pressure

newy_pt=newy_pre


X_train, X_test, y_train, y_test = train_test_split(
    newX, newy_pt, train_size=0.8, random_state=0)

modelp = Sequential()





#for pressure
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



#----------------------------------------------loss
plt.figure()

plt.plot(histp.history['loss'])
plt.plot(histp.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

#--------------------------------------------accura
#_--------------------------------------------------------input natural data

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


df_input['P/T']=df_input['Gpa.3']*1000/df_input['T']




print(df_input.columns)



df_input2=df_input[df_input['P/T']<=5]

pt_lee=df_input2['P/T']

y_lee_T=df_input2['T'].values
y_lee_P=df_input2['Gpa.3'].values

data_lee = np.zeros((len(pt_lee),10))  
for i in range(0,len(pt_lee)):  
 for j in range(0,10):  
   data_lee[i][j] = df_input2.iloc[i,j]  
   

#####model resutls
T_ML=model.predict(data_lee)
T_ML=T_ML.flatten()


P_ML=modelp.predict(data_lee)
P_ML=P_ML.flatten()



#T_P_C   PA1   PT_CA

y_ca_T=df_input2['T_P_C'].values
y_ca_P=df_input2['P_A1'].values


#T_P_A PA2   PT_AA


y_aa_T=df_input2['T_P_C'].values
y_aa_P=df_input2['P_A2'].values


####compare lee and ML results on lee data

#--------------------- version 1

plt.figure()

#pukrca model a  / Albarede   y_aa
plt.scatter(y_aa_T,y_aa_P,marker='^',facecolor='y',edgecolor='k',linewidths=0.2)


#pukrca model c  / Albarede   y_ca
plt.scatter(y_ca_T,y_ca_P,marker='d',facecolor='b',edgecolor='k',linewidths=0.5)



#results from Lee et al.2009
plt.scatter(y_lee_T,y_lee_P,marker='s',facecolor='g',edgecolor='k',linewidths=0.5)

#data from ML models
plt.errorbar(T_ML*1000,P_ML, xerr=36,yerr=0.2, fmt='o', mfc='r',mec='k',
         ecolor='k',elinewidth=0.5,capthick=0,capsize=0.5)

#we need change xrre and yerr based on the rmse


plt.legend(['P Albarede,T Putirka A ','P Albarede,T Putirka C','Lee et al. 2009','this study'],loc='lower left',fontsize=10)

#plt.legend(['P Albarede,T Putirka C ','Lee et al. 2009','this study'],loc='lower left',fontsize=10)

ax = plt.gca()

plt.xlim(1300,1750)
plt.ylim(0,6)

ax.invert_yaxis()
plt.ylabel('Pressure (GPa)')
ax.xaxis.set_ticks_position('top')  #将x轴的位置设置在顶部

#ax.set_xticklabels(row_labels, minor=False)

ax.set_xlabel('Temperature (℃)')    
ax.xaxis.set_label_position('top') 


plt.savefig('P_and_T_natrualsample.png',dpi=300)
#------------------------version 2

   


#calcualte the P/T ratio
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

plt.figure()

pt_aa=1000*y_aa_P/y_aa_T
pt_ca=1000*y_ca_P/y_ca_T
pt_lee=1000*y_lee_P/y_lee_T
pt_ML=P_ML/T_ML


n, bins, patches = plt.hist(pt_aa,15, density=False,facecolor='r', edgecolor='r', alpha=0.8,linewidth = 1)
n, bins, patches = plt.hist(pt_ca,15, density=False,facecolor='y', edgecolor='y', alpha=0.8,linewidth = 1)
n, bins, patches = plt.hist(pt_lee, 15,density=False, facecolor='b',edgecolor='g', alpha=0.8,linewidth = 1)
n, bins, patches = plt.hist(pt_ML,15, density=False,facecolor='k', edgecolor='k', alpha=0.8,linewidth = 2)





#------------------------------------calculate the P and T for emeishan LIP
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------
data_emeishan = pd.read_excel('emeishan.xlsx',header = None,skipfooter= 1,index_col=1)

#train data determined from dataframe
Emeishandata = np.zeros((1075,10))  
for i in range(0,1075):  
 for j in range(0,10):  
   Emeishandata[i][j] = data_emeishan.iloc[i+1,j+3]  
   
#####model resutls
T_Emei=model.predict(Emeishandata)
T_Emei=T_Emei.flatten()


P_Emei=modelp.predict(Emeishandata)
P_Emei=P_Emei.flatten()


#--------------------- emeishan

plt.figure()



#data from ML models
plt.errorbar(T_Emei*1000,P_Emei, xerr=36,yerr=0.2, fmt='o', mfc='r',mec='k',ecolor='k',elinewidth=0.5,capthick=0,capsize=0.5)


#--------------------- emeishan
#read the other points

df_emei= pd.read_excel('emei.xlsx',header = 0,skipfooter= 0,index_col=0)


p_emei_lee=df_emei['P'].values
t_emei_lee=df_emei['T'].values

plt.scatter(t_emei_lee,p_emei_lee,marker='s',facecolor='g',edgecolor='k',linewidths=0.5)

#we need change xrre and yerr based on the rmse

ax = plt.gca()

plt.xlim(1200,1750)
plt.ylim(0,6)

ax.invert_yaxis()
plt.ylabel('Pressure (GPa)')
ax.xaxis.set_ticks_position('top')  #将x轴的位置设置在顶部

#ax.set_xticklabels(row_labels, minor=False)

ax.set_xlabel('Temperature (℃)')    
ax.xaxis.set_label_position('top') 


#plt.savefig('P_and_T_Emeishan.png',dpi=300)







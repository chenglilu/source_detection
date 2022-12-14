#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 16:24:55 2021

@author: lilucheng
"""
#determine P and T not P/T ratio 
#compare to lee et al.2009 and previous




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
                 batch_size=20, epochs=300,
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

#--------------------------------------------accurancy



T_ML=model.predict(newX)
T_ML=T_ML.flatten()






P_ML=modelp.predict(newX)
P_ML=P_ML.flatten()

#up: prepare two models
#------------------------------------------------------------
#below:prepare lee et al.2009







#--------------------------------------------------------------------------------lee et al.

data = pd.read_excel(
    'data2_check_dry.xlsx', header=0, skipfooter=1, index_col=0)



df_peri=data[data['group']==1]
df_peri_dry=df_peri[df_peri['hydrous or not']==1]
df_peri_dry['H2O ']=0

df=df_peri_dry.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,30]]

#sio2: 60.08  ,  tio2: 79.87,   al2o3: 101.96  ,  cr2o3: 151.99,    feo: 71.85
#MnO: 86.94,    MgO: 40.3,   CaO: 56.08,   Na2O:  61.98,  K2O:94.2 H2O: 18.02
df['SiO2m']=df['SiO2 ']/60.08
df['TiO2m']=df['TiO2 ']/79.87
df['Al2O3m']=df['Al2O3 ']/101.96
df['Cr2O3m']=df['Cr2O3 ']/151.99
df['FeOm']=df['FeO ']/71.85
df['MnOm']=df['MnO ']/86.94
df['MgOm']=df['MgO ']/40.3
df['CaOm']=df['CaO ']/56.08
df['Na2Om']=df['Na2O ']/61.98
df['K2Om']=df['K2O ']/94.2
df['H2Om']=df['H2O ']/18.02

df.columns.get_loc('SiO2m')
df.columns.get_loc('H2Om')


df['molesum']=df.iloc[:,18:28].sum(axis=1)




df['SiO2']=100*df['SiO2m']/df['molesum']
df['TiO2']=100*df['TiO2m']/df['molesum']
df['Al2O3']=100*df['Al2O3m']/df['molesum']
df['Cr2O3']=100*df['Cr2O3m']/df['molesum']
df['FeO']=100*df['FeOm']/df['molesum']
df['MnO']=100*df['MnOm']/df['molesum']
df['MgO']=100*df['MgOm']/df['molesum']
df['CaO']=100*df['CaOm']/df['molesum']
df['Na2O']=100*df['Na2Om']/df['molesum']
df['K2O']=100*df['K2Om']/df['molesum']
df['H2O']=100*df['H2Om']/df['molesum']



####calculate single and sum



df['Si']=df['SiO2']
df['Ti']=df['TiO2']
df['Al']=2*df['Al2O3']
df['Cr']=2*df['Cr2O3']
df['Fe']=df['FeO']
df['Mn']=df['MnO']
df['Mg']=df['MgO']
df['Ca']=df['CaO']
df['Na']=2*df['Na2O']
df['K']=2*df['K2O']
df['H']=2*df['H2O']

df.columns.get_loc('Si')
df.columns.get_loc('H')


df['singlesum']=df.iloc[:,41:51].sum(axis=1)



df['Si%']=100*df['Si']/df['singlesum']
df['Ti%']=100*df['Ti']/df['singlesum']
df['Al%']=100*df['Al']/df['singlesum']
df['Cr%']=100*df['Cr']/df['singlesum']
df['Fe%']=100*df['Fe']/df['singlesum']
df['Mn%']=100*df['Mn']/df['singlesum']
df['Mg%']=100*df['Mg']/df['singlesum']
df['Ca%']=100*df['Ca']/df['singlesum']
df['Na%']=100*df['Na']/df['singlesum']
df['K%']=100*df['K']/df['singlesum']
df['H%']=100*df['H']/df['singlesum']


#molecular

df['Si4O8']=0.25*(df['Si%']-0.5*(df['Fe%']+df['Mg%']+df['Ca%'])-0.5*df['Na%']-0.5*df['K%'])
df['Ti4O8']=0.25*df['Ti%']
df['Al16/3O8']=0.375*(0.5*df['Al%']-0.5*df['Na%'])
df['Cr16/3O8']=0.375*df['Cr%']
df['Fe4Si2O8']=0.25*df['Fe']
df['Mn4Si2O8']=0.25*df['Mn']
df['Mg4Si2O8']=0.25*df['Mg']
df['Ca4Si2O8']=0.25*df['Ca']
df['Na2Al2Si2O8']=0.5*df['Na%']
df['K2Al2Si2O8']=0.5*df['K%']
df['H16O8']=0.125*df['H%']


df.columns.get_loc('Si4O8')
df.columns.get_loc('H16O8')


df['molecularsum']=df.iloc[:,64:74].sum(axis=1)

#change into percentage

df['Si4O8%']=100*df['Si4O8']/df['molecularsum']
df['Ti4O8%']=100*df['Ti4O8']/df['molecularsum']
df['Al16/3O8%']=100*df['Al16/3O8']/df['molecularsum']
df['Cr16/3O8%']=100*df['Cr16/3O8']/df['molecularsum']
df['Fe4Si2O8%']=100*df['Fe4Si2O8']/df['molecularsum']
df['Mn4Si2O8%']=100*df['Mn4Si2O8']/df['molecularsum']
df['Mg4Si2O8%']=100*df['Mg4Si2O8']/df['molecularsum']
df['Ca4Si2O8%']=100*df['Ca4Si2O8']/df['molecularsum']
df['Na2Al2Si2O8%']=100*df['Na2Al2Si2O8']/df['molecularsum']
df['K2Al2Si2O8%']=100*df['K2Al2Si2O8']/df['molecularsum']
df['H16O8%']=100*df['H16O8']/df['molecularsum']




Mg4Si2O8=df['Mg4Si2O8%'].to_numpy()
Si4O8=df['Si4O8%'].to_numpy()
H16O8=df['H16O8%'].to_numpy()
Fe4Si2O8=df['Fe4Si2O8%'].to_numpy()
Ca4Si2O8=df['Ca4Si2O8%'].to_numpy()



T=916.45+13.68*Mg4Si2O8+(4580/Si4O8)-0.509*H16O8*Mg4Si2O8


Tk=T+273.15   #change temperature in Kelvin

P=(np.log(Si4O8)-4.019+0.0165*(Fe4Si2O8)+0.0005*(Ca4Si2O8)**2)/(-770*Tk**(-1)+0.0058*Tk**(1/2)-0.003*H16O8)


#plot Temperature
True_T=df['T (??C) '].to_numpy()


#plot Pressure
True_P=df['P (GPa) '].to_numpy()




#two figure
#one temperature:
    

plt.figure()

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.subplots_adjust(left=0.1, right=0.95,bottom=0.12,top=0.95, wspace=0.2, hspace=0.2)


plt.scatter(True_T,(T-True_T)/True_T,marker='o',facecolor='m',edgecolor='k')   #lee  temperature

plt.scatter(True_T,(T_ML*1000-True_T)/True_T,marker='o',facecolor='g',edgecolor='k')   #ML  temrature

plt.legend(['Lee et al. 2009','this study'])
#plt.text(1200,-0.1,'Peridotitic',fontsize=12)
#plt.text(0.3,2.5,'One red point is out of the range',fontsize=10)


plt.plot([900,2000],[0,0],'r-')
plt.xlim(900,2000)
plt.ylim(-0.08,0.20)
plt.xlabel('Experimental T (???)',fontsize=12)
plt.ylabel('\u0394T (???)/Exprimental T',fontsize=12)


#two pressure:
    

plt.subplot(1,2,2)

plt.scatter(True_P,(P-True_P),marker='o',facecolor='m',edgecolor='k')   #lee  pressure

plt.scatter(True_P,(P_ML-True_P),marker='o',facecolor='g',edgecolor='k')   #ML  pressure

#plt.legend(['Lee et al. 2009','this study'])
#plt.text(1.8,-2.8,'Peridotitic',fontsize=12)
#plt.text(0.3,2.5,'One red point is out of the range',fontsize=10)


plt.plot([0,8],[0,0],'r-')
plt.xlim(0,8)
plt.ylim(-1.5,3)
plt.xlabel('Experimental P (GPa)',fontsize=12)
plt.ylabel('\u0394P (GPa)',fontsize=12)

plt.savefig('T_and_P_compare_lee.png',dpi=300)

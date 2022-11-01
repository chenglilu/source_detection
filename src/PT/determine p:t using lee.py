#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 13:57:05 2021

@author: lilucheng
"""
#basic equation
#Si4O8=0.25*(SiO2-0.5*(Feo+MgO+CaO)-Na2o-K2o)
#Ti4o8=0.25*Tio2
#Al16/3O8=0.375*(Al2O3-Na2o)
#Cr16/3O8=0.375*(Cr2O3)
#Fe16/3O8=0.375*(Fe2O3)
#Fe4Si2O8=0.25*Feo
#Mg4Si2O8=0.25*Mgo
#Ca4Si2O8=0.25*CaO
#Na2Al2Si2O8=Na2O
#K2Al2Si2O8=K2O
#P16/5O8=0.625*P2O5
#H16O8=0.125*H2O


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel(
    'data2_check_dry.xlsx', header=0, skipfooter=1, index_col=0)



df_peri=data[data['group']==1]
df_peri_dry=df_peri[df_peri['hydrous or not']==1]
df_peri_dry['H2O ']=0





Mg4Si2O8=df_peri_dry['MgO '].to_numpy()*0.25
Si4O8=0.25*(df_peri_dry['SiO2 '].to_numpy()-0.5*(df_peri_dry['FeO '].to_numpy()+df_peri_dry['MgO '].to_numpy()
                                                 +df_peri_dry['CaO '].to_numpy())-df_peri_dry['Na2O '].to_numpy()-df_peri_dry['K2O '].to_numpy())

H16O8=df_peri_dry['H2O '].to_numpy()*0.125
Fe4Si2O8=df_peri_dry['FeO '].to_numpy()*0.25
Ca4Si2O8=df_peri_dry['CaO '].to_numpy()*0.25



T=916.45+13.68*Mg4Si2O8+(4580/Si4O8)-0.509*H16O8*Mg4Si2O8


Tk=T+273.15   #change temperature in Kelvin

P=(np.log(Si4O8)-4.019+0.0165*(Fe4Si2O8)+0.0005*(Ca4Si2O8)**2)/(-770*Tk**(-1)+0.0058*Tk**(1/2)-0.003*H16O8)

True_T=df_peri_dry['T (°C) '].to_numpy()

#plot Temperature
plt.figure()
plt.scatter(True_T,T,marker='o',facecolor='g',edgecolor='k')
plt.plot([1000,2000],[1000,2000],'r-')  #pt
plt.xlim(1000,2000)
plt.ylim(1000,2000)
plt.ylabel('Predected T (℃)',fontsize=12)
plt.xlabel('True T (℃)',fontsize=12)


#plot Pressure
True_P=df_peri_dry['P (GPa) '].to_numpy()
plt.figure()
plt.scatter(True_P,P,marker='o',facecolor='g',edgecolor='k')
#plt.plot([0,7],[0,7],'r-')  #pt
#plt.xlim(0,7)
#plt.ylim(0,7)
plt.ylabel('Predected P (GPa)',fontsize=12)
plt.xlabel('True P (GPa)',fontsize=12)


#plot P/T
True_P=df_peri_dry['P (GPa) '].to_numpy()
True_PT=True_P*1000/True_T
PT=P*1000/T

plt.figure()
plt.scatter(True_PT,PT,marker='o',facecolor='g',edgecolor='k')
#plt.plot([0,7],[0,7],'r-')  #pt
#plt.xlim(0,7)
#plt.ylim(0,7)
plt.ylabel('Predected P (GPa)',fontsize=12)
plt.xlabel('True P (GPa)',fontsize=12)

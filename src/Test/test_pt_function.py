#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 10:06:07 2021

@author: lilucheng
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math



df= pd.read_excel(
    'test_pt.xlsx', header=0, skipfooter=0, index_col=0)


Mg4Si2O8=df['2Mg2SiO4'].to_numpy()
Si4O8=df['4SiO2'].to_numpy()
H16O8=df['8H2O'].to_numpy()
Fe4Si2O8=df['2Fe2SiO4'].to_numpy()
Ca4Si2O8=df['2Ca2SiO4'].to_numpy()



T=916.45+13.68*Mg4Si2O8+(4580/Si4O8)-0.509*H16O8*Mg4Si2O8


Tk=T+273.15   #change temperature in Kelvin

P=(np.log(Si4O8)-4.019+0.0165*(Fe4Si2O8)+0.0005*(Ca4Si2O8)**2)/(-770*Tk**(-1)+0.0058*Tk**(1/2)-0.003*H16O8)


#plot Temperature
True_T=df['T'].to_numpy()
plt.figure()
plt.scatter(True_T,T,marker='o',facecolor='g',edgecolor='k')
plt.plot([1000,2000],[1000,2000],'r-')  #pt
#plt.xlim(1000,2000)
#plt.ylim(1000,2000)
plt.ylabel('Predected T (℃)',fontsize=12)
plt.xlabel('True T (℃)',fontsize=12)


my=sum(True_T)/len(True_T)

r2_T=1-sum((T-True_T)**2)/sum((True_T-my)**2)



rmse_T=math.sqrt(sum((True_T-T)**2)/len(True_T))


name=('R2=%6.2f\nRMSE =%6.2f' %(r2_T,rmse_T))


plt.title(name)





#plot Pressure
True_P=df['Gpa'].to_numpy()
plt.figure()
plt.scatter(True_P,P,marker='o',facecolor='g',edgecolor='k')
plt.plot([0,7],[0,7],'r-')  #pt
plt.xlim(0,7)
plt.ylim(0,7)
plt.ylabel('Predected P (GPa)',fontsize=12)
plt.xlabel('True P (GPa)',fontsize=12)


#plot P/T
True_P=df['Gpa'].to_numpy()
True_PT=True_P*1000/True_T
PT=P*1000/T

plt.figure()
plt.scatter(True_PT,PT,marker='o',facecolor='g',edgecolor='k')
plt.plot([0,7],[0,7],'r-')  #pt
plt.xlim(0,7)
plt.ylim(0,7)
plt.ylabel('Predected P/T (GPa)',fontsize=12)
plt.xlabel('True P/T (GPa)',fontsize=12)


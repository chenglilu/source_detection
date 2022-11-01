#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:57:44 2021

@author: lilucheng
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math



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
True_T=df['T (°C) '].to_numpy()
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
name=('${R^2}$=%6.2f\nRMSE =%6.2f' %(r2_T,rmse_T))
plt.title(name)





#plot Pressure
True_P=df['P (GPa) '].to_numpy()
plt.figure()
plt.scatter(True_P,P,marker='o',facecolor='g',edgecolor='k')
plt.plot([0,7],[0,7],'r-')  #pt
plt.xlim(0,7)
plt.ylim(0,7)
plt.ylabel('Predected P (GPa)',fontsize=12)
plt.xlabel('True P (GPa)',fontsize=12)



#df2=pd.DataFrame(True_P,P)



my=sum(True_P)/len(True_P)
r2_P=1-sum((P-True_P)**2)/sum((True_P-my)**2)
rmse_P=math.sqrt(sum((True_P-P)**2)/len(True_P))
name=('${R^2}$=%6.2f\nRMSE =%6.2f' %(r2_P,rmse_P))

plt.title(name)




#plot P/T
True_P=df['P (GPa) '].to_numpy()
True_PT=True_P*1000/True_T
PT=P*1000/T

plt.figure()
plt.scatter(True_PT,PT,marker='o',facecolor='g',edgecolor='k')
plt.plot([0,6],[0,6],'r-')  #pt
plt.xlim(0,6)
plt.ylim(0,6)
plt.ylabel('Predected P/T (MPa/℃)',fontsize=12)
plt.xlabel('True P/T (MPa/℃)',fontsize=12)

my=sum(True_PT)/len(True_PT)
r2_PT=1-sum((PT-True_PT)**2)/sum((True_PT-my)**2)
rmse_PT=math.sqrt(sum((True_PT-PT)**2)/len(True_PT))
name=('${R^2}$=%6.2f\nRMSE =%6.2f' %(r2_PT,rmse_PT))

plt.text(4,2,name,fontsize=12)
plt.text(4,3,'Lee et al. 2009',fontsize=12)

plt.title('Peridotitic')

plt.savefig('peridotites_lee.png',dpi=300)


#------------------------------------------------------------------------add our method points


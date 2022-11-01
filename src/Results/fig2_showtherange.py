#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 19:48:27 2021

@author: lilucheng
"""


import pandas as pd

import matplotlib.pyplot as plt

#-------------------------------------------------------
# Read the file  dataframe
df = pd.read_excel('data2_check.xlsx',header = 0,skipfooter= 1,index_col=0)

df2=df[['P (GPa) ','T (°C) ','group']]

df2= df2.rename(columns = {'P (GPa) ': 'P', 'T (°C) ': 'T'})

#df2['P (GPa) '].hist(by=df['group'])



#print(df2.groupby(['group']).mean())
#grouped = df2.groupby('group')


df_p=df2[df2['group']==1]
df_t=df2[df2['group']==2]
df_m=df2[df2['group']==3]


plt.figure(figsize=(10,4))

plt.subplots_adjust(left=0.1, right=0.9,bottom=0.12,top=0.95, wspace=0.3, hspace=0.3)

plt.subplot(1,3,1)
plt.hist(df_p['T'], 20,density=False, facecolor='limegreen',edgecolor='limegreen', alpha=1,histtype='step')
plt.hist(df_t['T'],20, density=False,facecolor='deepskyblue', edgecolor='r', alpha=1,histtype='step')
plt.hist(df_m['T'],20, density=False, facecolor='0.5',edgecolor='k', alpha=1,histtype='step')

plt.text(450,82,'(a)',fontsize=14)

plt.xlabel('T(℃)',fontsize=12) 
#plt.ylabel('Probability')
plt.ylabel('Number',fontsize=12)  


plt.subplot(1,3,2)
plt.hist(df_p['P'], 10,density=False, facecolor='limegreen',edgecolor='limegreen', alpha=1,histtype='step')
plt.hist(df_t['P'],10, density=False,facecolor='deepskyblue', edgecolor='r', alpha=1,histtype='step')
plt.hist(df_m['P'],10, density=False, facecolor='0.5',edgecolor='k', alpha=1,histtype='step')
plt.legend(['Peridotitic','Transitional','Mafic'],loc='upper right',fontsize=10)

plt.text(-4,192,'(b)',fontsize=14)

#plt.legend(['Peridotitic','Transitional','Mafic'],loc='upper right')
plt.xlabel('P(GPa)',fontsize=12) 
#plt.ylabel('Probability')
plt.ylabel('Number',fontsize=12)  



plt.subplot(1,3,3)
plt.hist(df_p['P']*1000/df_p['T'], 10,density=False, facecolor='limegreen',edgecolor='limegreen', alpha=1,histtype='step')
plt.hist(df_t['P']*1000/df_t['T'],10, density=False,facecolor='deepskyblue', edgecolor='r', alpha=1,histtype='step')
plt.hist(df_m['P']*1000/df_m['T'],10, density=False, facecolor='0.5',edgecolor='k', alpha=1,histtype='step')
#plt.legend(['Peridotitic','Transitional','Mafic'],loc='upper right',fontsize=12)

plt.text(-2,190,'(c)',fontsize=14)

#plt.legend(['Peridotitic','Transitional','Mafic'],loc='upper right')
plt.xlabel('P/T (MPa/℃)',fontsize=12) 
#plt.ylabel('Probability')
plt.ylabel('Number',fontsize=12)  

plt.savefig('fig2_pt_range.png', dpi=300)




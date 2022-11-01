#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 09:30:51 2021

@author: lilucheng
"""
#plot accuracy results

import matplotlib.pyplot as plt
import numpy as np


x1=[1,2,3,4,5,6]
lr=[75,	74,75,81,86,5]
nb=[72,72,72,77,80,20]
rf=[77,75,76,77	,92	,0]
svm=[78,76,77,87,87,0]
ann=[89,88,89,92,93,60]
ann_ratio=[84,82,82	,88	,93	,49]


x2=[3,4,5]
camg=[58,69,63,0]
fc3ms=[51,99,31,0]

x3=[3,4,5,6]
fcms=[61,88,54,13]


plt.figure
plt.plot(x1,lr,color=[0.5,0.8,0],linewidth=0.8)
plt.plot(x1,nb,color=[0.4,0.8,0],linewidth=0.8)
plt.plot(x1,rf,color=[0.3,0.8,0],linewidth=0.8)
plt.plot(x1,svm,color=[0.2,0.8,0],linewidth=0.8)
plt.plot(x1,ann,color=[0.7,0,0],linewidth=1)
#plt.plot(x1,ann_ratio,color=[0.8,0,0],linewidth=1)


plt.plot(x3,camg,color='0.5',linewidth=0.8)
plt.plot(x3,fc3ms,color='0.4',linewidth=0.8)
plt.plot(x3,fcms,color='0.2',linewidth=0.8)

plt.text(0.6,86,'ANN',fontsize=8)
plt.text(0.6,80,'SVM',fontsize=8)
plt.text(0.7,76,'RF',fontsize=8)
plt.text(0.7,72,'LR',fontsize=8)
plt.text(0.7,68,'NB',fontsize=8)



plt.text(2,63,'FCKANTMS',fontsize=8)
plt.text(2.1,55,'CaO-MgO',fontsize=8)
plt.text(2.3,47,'FC3MS',fontsize=8)

xx=[0,8]
l1=[90,90]
plt.plot(xx,l1,'--',color='0.5',linewidth=0.8)
plt.text(6.2,91,'90%',fontsize=7)


plt.xlim(0.5,6.5)

plt.xticks(x1, ['A_train','A_test','A_overall','A_perido','A_mafic','A_trans'])
plt.ylabel('Accuracy(%)')

plt.savefig('total accuracy',dpi=300)
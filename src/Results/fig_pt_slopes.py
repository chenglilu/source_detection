#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:13:21 2021

@author: lilucheng
"""
#draw another p/t figure using P and T slopes
#data: y_ha   y_em


#the maxiumu tempearture for emeishan could be 1590
#the maximum temperature for hawwiio  could be 1400-1500



import random


#tem_em=np.random.uniform(1450,1600,size=(len(y_em),1))
tem_em=1590
pressure_em=tem_em*y_em

#pressure_em=np.sort(tem_em)*np.sort(y_em)

pressure_em=pressure_em/1000



tem_ha=1500
pressure_ha=tem_ha*tem_ha
pressure_ha=pressure_ha/1000



plt.figure()
ax=plt.gca()



plt.scatter(tem_em,pressure_em)

plt.xlim(1000,2000)


ax.xaxis.set_ticks_position('top')  #将x轴的位置设置在顶部

plt.gca().invert_yaxis() 

plt.xlabel('Temperature(℃)')
plt.ylabel('Pressure(GPa)')
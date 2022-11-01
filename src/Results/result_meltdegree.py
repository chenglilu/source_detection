#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 12:00:30 2021

@author: lilucheng
"""

#results to show accuarcy vs melt degree, pressure, tempature



import pandas as pd
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from numpy import *
from sklearn.metrics import mean_squared_error 


#-------------------------------------------------------
# Read the file  dataframe
data = pd.read_excel('data2_check.xlsx',header = None,skipfooter= 1,index_col=1)

#change into the data we need float
#train data determined from dataframe
Traindata = np.zeros((915,10))  
for i in range(0,915):  
 for j in range(0,10):  
   Traindata[i][j] = data.iloc[i+1,j+6]  


#change nan into 0
for i in range(0,915):  
 for j in range(0,10):  
  if (np.isnan(Traindata[i][j])):
      Traindata[i][j]= 0
  
   
   
   

#lable from dataframe
Group=np.zeros((915,1))
for i in range(0,915):   
   Group[i] = data.iloc[i+1,24]  


#-------------------------------------------------------
X = Traindata
y = Group


newX=X
newy=y

X_train, X_test, y_train, y_test = train_test_split(newX, newy, train_size=0.8, random_state = 0)


clf = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 20), random_state=1)


clf = clf.fit(X_train, y_train)


#------------------------------------------------------------------------------plot

   
y_result=clf.predict(X)

y_result=y_result.reshape(-1,1)


difference_total=np.zeros((915,1))
difference_total=y-y_result


idx=np.where(difference_total==0)  
idx0=idx[0]
Same_rate=len(idx0)/len(y_result)
print('Same number of all the data',Same_rate)   




#-----peridotite right and wrong
idx1r=np.where((y==1) & (difference_total==0))
idx1w=np.where((y==1) & (difference_total!=0))

idx1r0=idx1r[0]
idx1w0=idx1w[0]

#-----transitional right and wrong

idx2r=np.where((y==2) & (difference_total==0))
idx2w=np.where((y==2) & (difference_total!=0))
idx2r0=idx2r[0]
idx2w0=idx2w[0]

#-----mafic right and wrong


idx3r=np.where((y==3) & (difference_total==0))
idx3w=np.where((y==3) & (difference_total!=0))

idx3r0=idx3r[0]
idx3w0=idx3w[0]

#-------------------------------------------------------
#prepare meltdegree,temperature,pressure

meltdegree=np.zeros((915,1))
for i in range(0,915):   
   meltdegree[i] = data.iloc[i+1,3]  

temperature=np.zeros((915,1))
for i in range(0,915):   
   temperature[i] = data.iloc[i+1,2]  

pressure=np.zeros((915,1))
for i in range(0,915):   
   pressure[i] = data.iloc[i+1,1]  

#############################
#############################
#########start select################
#############################
#--------------------------------------------------------------------------------------peridotite  
#-------------------------------------------------------
#pressure

p0_p=np.where((y==1) & (pressure>=0) & (pressure<2))   #total
p0_total=len(p0_p[0])
 
p0_p_r=np.where((y==1) & (pressure>=0) & (pressure<2) & (difference_total==0))    #right
p0_right=len(p0_p_r[0])  


if p0_total==0:
    p0_ratio_p=0
else:
    p0_ratio_p=p0_right/p0_total*100
#-------------   

p1_p=np.where((y==1) & (pressure>=2) & (pressure<4))   #total
p1_total=len(p1_p[0])
 
p1_p_r=np.where((y==1) & (pressure>=2) & (pressure<4) & (difference_total==0))    #right
p1_right=len(p1_p_r[0])   


if p1_total==0:
    p1_ratio_p=0
else:
    p1_ratio_p=p1_right/p1_total*100
#-------------

p2_p=np.where((y==1) & (pressure>=4) & (pressure<6))   #total
p2_total=len(p2_p[0])
 
p2_p_r=np.where((y==1) & (pressure>=4) & (pressure<6) & (difference_total==0))    #right
p2_right=len(p2_p_r[0])  

if p2_total==0:
    p2_ratio_p=0
else:
    p2_ratio_p=p2_right/p2_total*100
#-------------

p3_p=np.where((y==1) & (pressure>=6) & (pressure<8))   #total
p3_total=len(p3_p[0])
 
p3_p_r=np.where((y==1) & (pressure>=6) & (pressure<8) & (difference_total==0))    #right
p3_right=len(p3_p_r[0])  

if p3_total==0:
    p3_ratio_p=0
else:
    p3_ratio_p=p3_right/p3_total*100

#-------------

p4_p=np.where((y==1) & (pressure>=8))   #total
p4_total=len(p4_p[0])
 
p4_p_r=np.where((y==1) & (pressure>=8) & (difference_total==0))    #right
p4_right=len(p4_p_r[0])  

if p4_total==0:
    p4_ratio_p=0
else:
    p4_ratio_p=p4_right/p4_total*100

#-------------
    

#temperature

t0_p=np.where((y==1) & (temperature<=1200) )   #total
t0_total=len(t0_p[0])
 
t0_p_r=np.where((y==1) & (temperature<1200) & (difference_total==0))    #right
t0_right=len(t0_p_r[0])  

if t0_total==0:
    t0_ratio_p=0
else:
    t0_ratio_p=t0_right/t0_total*100
#-------------   

t1_p=np.where((y==1) & (temperature>=1200) & (temperature<1400))   #total
t1_total=len(t1_p[0])
 
t1_p_r=np.where((y==1) & (temperature>=1200) & (temperature<1400) & (difference_total==0))    #right
t1_right=len(t1_p_r[0])   

if t1_total==0:
    t1_ratio_p=0
else:
    t1_ratio_p=t1_right/t1_total*100
#-------------

t2_p=np.where((y==1) & (temperature>=1400) & (temperature<1600))   #total
t2_total=len(t2_p[0])
 
t2_p_r=np.where((y==1) & (temperature>=1400) & (temperature<1600) & (difference_total==0))    #right
t2_right=len(t2_p_r[0])  

if t2_total==0:
    t2_ratio_p=0
else:
    t2_ratio_p=t2_right/t2_total*100
#-------------

t3_p=np.where((y==1) & (temperature>=1600) & (temperature<1800))   #total
t3_total=len(t3_p[0])
 
t3_p_r=np.where((y==1) & (temperature>=1600) & (temperature<1800) & (difference_total==0))    #right
t3_right=len(t3_p_r[0])  

if t3_total==0:
    t3_ratio_p=0
else:
    t3_ratio_p=t3_right/t3_total*100

#-------------

t4_p=np.where((y==1) & (temperature>=1800))   #total
t4_total=len(t4_p[0])
 
t4_p_r=np.where((y==1) & (temperature>=1800) & (difference_total==0))    #right
t4_right=len(t4_p_r[0])  

if t4_total==0:
    t4_ratio_p=0
else:
    t4_ratio_p=t4_right/t4_total*100

#-------------    
    
 
    
#-------------------------------------------------------
#meltdegree

m0_p=np.where((y==1) & (meltdegree<=0.2) )   #total
m0_total=len(m0_p[0])
 
m0_p_r=np.where((y==1) & (meltdegree<0.2) & (difference_total==0))    #right
m0_right=len(m0_p_r[0])  

if m0_total==0:
    m0_ratio_p=0
else:
    m0_ratio_p=m0_right/m0_total*100
#-------------   

m1_p=np.where((y==1) & (meltdegree>=0.2) & (meltdegree<0.4))   #total
m1_total=len(m1_p[0])
 
m1_p_r=np.where((y==1) & (meltdegree>=0.2) & (meltdegree<0.4) & (difference_total==0))    #right
m1_right=len(m1_p_r[0])   

if m1_total==0:
    m1_ratio_p=0
else:
    m1_ratio_p=m1_right/m1_total*100
#-------------

m2_p=np.where((y==1) & (meltdegree>=0.4) & (meltdegree<0.6))   #total
m2_total=len(m2_p[0])
 
m2_p_r=np.where((y==1) & (meltdegree>=0.4) & (meltdegree<0.6) & (difference_total==0))    #right
m2_right=len(m2_p_r[0])  

if m2_total==0:
    m2_ratio_p=0
else:
    m2_ratio_p=m2_right/m2_total*100
#-------------

m3_p=np.where((y==1) & (meltdegree>=0.6) & (meltdegree<0.8))   #total
m3_total=len(m3_p[0])
 
m3_p_r=np.where((y==1) & (meltdegree>=0.6) & (meltdegree<0.8) & (difference_total==0))    #right
m3_right=len(m3_p_r[0])  

if m3_total==0:
    m3_ratio_p=0
else:
    m3_ratio_p=m3_right/m3_total*100

#-------------

m4_p=np.where((y==1) & (meltdegree>=0.8))   #total
m4_total=len(m4_p[0])
 
m4_p_r=np.where((y==1) & (meltdegree>=0.8) & (difference_total==0))    #right
m4_right=len(m4_p_r[0])  

if m4_total==0:
    m4_ratio_p=0
else:
    m4_ratio_p=m4_right/m4_total*100

#-------------    

#-----------------------------------------------------------------------------------------------------transitional    
#-------------------------------------------------------
#pressure

p0_t=np.where((y==2) & (pressure>=0) & (pressure<2))   #total
p0_total=len(p0_t[0])
 
p0_t_r=np.where((y==2) & (pressure>=0) & (pressure<2) & (difference_total==0))    #right
p0_right=len(p0_t_r[0])  

if p0_total==0:
    p0_ratio_t=0
else:
    p0_ratio_t=p0_right/p0_total*100
#-------------   

p1_t=np.where((y==2) & (pressure>=2) & (pressure<4))   #total
p1_total=len(p1_t[0])
 
p1_t_r=np.where((y==2) & (pressure>=2) & (pressure<4) & (difference_total==0))    #right
p1_right=len(p1_t_r[0])   

if p1_total==0:
    p1_ratio_t=0
else:
    p1_ratio_t=p1_right/p1_total*100
#-------------

p2_t=np.where((y==2) & (pressure>=4) & (pressure<6))   #total
p2_total=len(p2_t[0])
 
p2_t_r=np.where((y==2) & (pressure>=4) & (pressure<6) & (difference_total==0))    #right
p2_right=len(p2_t_r[0])  

if p2_total==0:
    p2_ratio_t=0
else:
    p2_ratio_t=p2_right/p2_total*100
#-------------

p3_t=np.where((y==2) & (pressure>=6) & (pressure<8))   #total
p3_total=len(p3_t[0])
 
p3_t_r=np.where((y==2) & (pressure>=6) & (pressure<8) & (difference_total==0))    #right
p3_right=len(p3_t_r[0])  

if p3_total==0:
    p3_ratio_t=0
else:
    p3_ratio_t=p3_right/p3_total*100

#-------------

p4_t=np.where((y==2) & (pressure>=8))   #total
p4_total=len(p4_t[0])
 
p4_t_r=np.where((y==2) & (pressure>=8) & (difference_total==0))    #right
p4_right=len(p4_t_r[0])  

if p4_total==0:
    p4_ratio_t=0
else:
    p4_rati_to=p4_right/p4_total*100

#-------------
    
    
    
#temperature

t0_t=np.where((y==2) & (temperature<=1200) )   #total
t0_total=len(t0_t[0])
 
t0_t_r=np.where((y==2) & (temperature<1200) & (difference_total==0))    #right
t0_right=len(t0_t_r[0])  

if t0_total==0:
    t0_ratio_t=0
else:
    t0_ratio_t=t0_right/t0_total*100
#-------------   

t1_t=np.where((y==2) & (temperature>=1200) & (temperature<1400))   #total
t1_total=len(t1_t[0])
 
t1_t_r=np.where((y==2) & (temperature>=1200) & (temperature<1400) & (difference_total==0))    #right
t1_right=len(t1_t_r[0])   

if t1_total==0:
    t1_ratio_t=0
else:
    t1_ratio_t=t1_right/t1_total*100
#-------------

t2_t=np.where((y==2) & (temperature>=1400) & (temperature<1600))   #total
t2_total=len(t2_t[0])
 
t2_t_r=np.where((y==2) & (temperature>=1400) & (temperature<1600) & (difference_total==0))    #right
t2_right=len(t2_t_r[0])  

if t2_total==0:
    t2_ratio_t=0
else:
    t2_ratio_t=t2_right/t2_total*100
#-------------

t3_t=np.where((y==2) & (temperature>=1600) & (temperature<1800))   #total
t3_total=len(t3_t[0])
 
t3_t_r=np.where((y==2) & (temperature>=1600) & (temperature<1800) & (difference_total==0))    #right
t3_right=len(t3_t_r[0])  

if t3_total==0:
    t3_ratio_t=0
else:
    t3_ratio_t=t3_right/t3_total*100

#-------------

t4_t=np.where((y==2) & (temperature>=1800))   #total
t4_total=len(t4_t[0])
 
t4_t_r=np.where((y==2) & (temperature>=1800) & (difference_total==0))    #right
t4_right=len(t4_t_r[0])  

if t4_total==0:
    t4_ratio_t=0
else:
    t4_ratio_t=t4_right/t4_total*100

#-------------    
    
 
    
#-------------------------------------------------------
#meltdegree

m0_t=np.where((y==2) & (meltdegree<=0.2) )   #total
m0_total=len(m0_t[0])
 
m0_t_r=np.where((y==2) & (meltdegree<0.2) & (difference_total==0))    #right
m0_right=len(m0_t_r[0])  

if m0_total==0:
    m0_ratio_t=0
else:
    m0_ratio_t=m0_right/m0_total*100
#-------------   

m1_t=np.where((y==2) & (meltdegree>=0.2) & (meltdegree<0.4))   #total
m1_total=len(m1_t[0])
 
m1_t_r=np.where((y==2) & (meltdegree>=0.2) & (meltdegree<0.4) & (difference_total==0))    #right
m1_right=len(m1_t_r[0])   

if m1_total==0:
    m1_ratio_t=0
else:
    m1_ratio_t=m1_right/m1_total*100
#-------------

m2_t=np.where((y==2) & (meltdegree>=0.4) & (meltdegree<0.6))   #total
m2_total=len(m2_t[0])
 
m2_t_r=np.where((y==2) & (meltdegree>=0.4) & (meltdegree<0.6) & (difference_total==0))    #right
m2_right=len(m2_t_r[0])  

if m2_total==0:
    m2_ratio_t=0
else:
    m2_ratio_t=m2_right/m2_total*100
#-------------

m3_t=np.where((y==2) & (meltdegree>=0.6) & (meltdegree<0.8))   #total
m3_total=len(m3_t[0])
 
m3_t_r=np.where((y==2) & (meltdegree>=0.6) & (meltdegree<0.8) & (difference_total==0))    #right
m3_right=len(m3_t_r[0])  

if m3_total==0:
    m3_ratio_t=0
else:
    m3_ratio_t=m3_right/m3_total*100

#-------------

m4_t=np.where((y==2) & (meltdegree>=0.8))   #total
m4_total=len(m4_t[0])
 
m4_t_r=np.where((y==2) & (meltdegree>=0.8) & (difference_total==0))    #right
m4_right=len(m4_t_r[0])  

if m4_total==0:
    m4_ratio_t=0
else:
    m4_ratio_t=m4_right/m4_total*100

#-------------    



#-----------------------------------------------------------------------------------mafic
#-------------------------------------------------------
#pressure

p0_m=np.where((y==3) & (pressure>=0) & (pressure<2))   #total
p0_total=len(p0_m[0])
 
p0_m_r=np.where((y==3) & (pressure>=0) & (pressure<2) & (difference_total==0))    #right
p0_right=len(p0_m_r[0])  

p0_ratio_m=p0_right/p0_total*100
#-------------   

p1_m=np.where((y==3) & (pressure>=2) & (pressure<4))   #total
p1_total=len(p1_m[0])
 
p1_m_r=np.where((y==3) & (pressure>=2) & (pressure<4) & (difference_total==0))    #right
p1_right=len(p1_m_r[0])   

p1_ratio_m=p1_right/p1_total*100
#-------------

p2_m=np.where((y==3) & (pressure>=4) & (pressure<6))   #total
p2_total=len(p2_m[0])
 
p2_m_r=np.where((y==3) & (pressure>=4) & (pressure<6) & (difference_total==0))    #right
p2_right=len(p2_m_r[0])  

p2_ratio_m=p2_right/p2_total*100
#-------------

p3_m=np.where((y==3) & (pressure>=6) & (pressure<8))   #total
p3_total=len(p3_m[0])
 
p3_m_r=np.where((y==3) & (pressure>=6) & (pressure<8) & (difference_total==0))    #right
p3_right=len(p3_m_r[0])  

if p3_total==0:
    p3_ratio_m=0
else:
    p3_ratio_m=p3_right/p3_total*100

#-------------

p4_m=np.where((y==3) & (pressure>=8))   #total
p4_total=len(p4_m[0])
 
p4_m_r=np.where((y==3) & (pressure>=8) & (difference_total==0))    #right
p4_right=len(p4_m_r[0])  

if p4_total==0:
    p4_ratio_m=0
else:
    p4_ratio_m=p4_right/p4_total*100

#-------------
    
    
    


#temperature

t0_m=np.where((y==3) & (temperature<=1200) )   #total
t0_total=len(t0_m[0])
 
t0_m_r=np.where((y==3) & (temperature<1200) & (difference_total==0))    #right
t0_right=len(t0_m_r[0])  

if t0_total==0:
    t0_ratio_m=0
else:
    t0_ratio_m=t0_right/t0_total*100
#-------------   

t1_m=np.where((y==3) & (temperature>=1200) & (temperature<1400))   #total
t1_total=len(t1_m[0])
 
t1_m_r=np.where((y==3) & (temperature>=1200) & (temperature<1400) & (difference_total==0))    #right
t1_right=len(t1_m_r[0])   

if t1_total==0:
    t1_ratio_m=0
else:
    t1_ratio_m=t1_right/t1_total*100
#-------------

t2_m=np.where((y==3) & (temperature>=1400) & (temperature<1600))   #total
t2_total=len(t2_m[0])
 
t2_m_r=np.where((y==3) & (temperature>=1400) & (temperature<1600) & (difference_total==0))    #right
t2_right=len(t2_m_r[0])  

if t2_total==0:
    t2_ratio_m=0
else:
    t2_ratio_m=t2_right/t2_total*100
#-------------

t3_m=np.where((y==3) & (temperature>=1600) & (temperature<1800))   #total
t3_total=len(t3_m[0])
 
t3_m_r=np.where((y==3) & (temperature>=1600) & (temperature<1800) & (difference_total==0))    #right
t3_right=len(t3_m_r[0])  

if t3_total==0:
    t3_ratio_m=0
else:
    t3_ratio_m=t3_right/t3_total*100

#-------------

t4_m=np.where((y==3) & (temperature>=1800))   #total
t4_total=len(t4_m[0])
 
t4_m_r=np.where((y==3) & (temperature>=1800) & (difference_total==0))    #right
t4_right=len(t4_m_r[0])  

if t4_total==0:
    t4_ratio_m=0
else:
    t4_ratio_m=t4_right/t4_total*100

#-------------    
    
 
    
#-------------------------------------------------------
#meltdegree

m0_m=np.where((y==3) & (meltdegree<=0.2) )   #total
m0_total=len(m0_m[0])
m0_total_m=m0_total
 
m0_m_r=np.where((y==3) & (meltdegree<0.2) & (difference_total==0))    #right
m0_right=len(m0_m_r[0])  

if m0_total==0:
    m0_ratio_m=0
else:
    m0_ratio_m=m0_right/m0_total*100
#-------------   

m1_m=np.where((y==3) & (meltdegree>=0.2) & (meltdegree<0.4))   #total
m1_total=len(m1_m[0])
 
m1_m_r=np.where((y==3) & (meltdegree>=0.2) & (meltdegree<0.4) & (difference_total==0))    #right
m1_right=len(m1_m_r[0])   

if m1_total==0:
    m1_ratio_m=0
else:
    m1_ratio_m=m1_right/m1_total*100
#-------------

m2_m=np.where((y==3) & (meltdegree>=0.4) & (meltdegree<0.6))   #total
m2_total=len(m2_m[0])
 
m2_m_r=np.where((y==3) & (meltdegree>=0.4) & (meltdegree<0.6) & (difference_total==0))    #right
m2_right=len(m2_m_r[0])  

if m2_total==0:
    m2_ratio_m=0
else:
    m2_ratio_m=m2_right/m2_total*100
#-------------

m3_m=np.where((y==3) & (meltdegree>=0.6) & (meltdegree<0.8))   #total
m3_total=len(m3_m[0])
 
m3_m_r=np.where((y==3) & (meltdegree>=0.6) & (meltdegree<0.8) & (difference_total==0))    #right
m3_right=len(m3_m_r[0])  

if m3_total==0:
    m3_ratio_m=0
else:
    m3_ratio_m=m3_right/m3_total*100

#-------------

m4_m=np.where((y==3) & (meltdegree>=0.8))   #total
m4_total=len(m4_m[0])
 
m4_m_r=np.where((y==3) & (meltdegree>=0.8) & (difference_total==0))    #right
m4_right=len(m4_m_r[0])  

if m4_total==0:
    m4_ratio_m=0
else:
    m4_ratio_m=m4_right/m4_total*100

#-------------       
    
    
    


#------------------------------------------------------------------------------------------figure
    
fig=plt.figure()   

plt.subplots_adjust(left=0.15, right=0.85,bottom=0.14,top=0.98, wspace=0.45, hspace=0)

#########----------------------------------
#########----------------------------------peridotite
plt.subplot(3,1,1)
#ax2 = fig.add_subplot(1,3,1)
#ax2.set_position([0.1,0.2,0.8,0.79])
#locate=[1,2,3,4,5]




#s=[p0_total,p1_total,p2_total,p3_total,p4_total]
plt.scatter([1, 2, 3, 4, 5],[m0_ratio_p,m1_ratio_p,m2_ratio_p,m3_ratio_p,m4_ratio_p],marker='s',c='orangered',edgecolors='k',s=50)  # 画散点图, alpha=0.6 表示不透明度为 0.6
plt.scatter([1, 2, 3, 4, 5],[t0_ratio_p,t1_ratio_p,t2_ratio_p,t3_ratio_p,t4_ratio_p],c='g',edgecolors='k',s=50)  # 画散点图, alpha=0.6 表示不透明度为 0.6

ln1=plt.scatter([1, 2, 3, 4, 5],[p0_ratio_p,p1_ratio_p,p2_ratio_p,p3_ratio_p,p4_ratio_p],marker='^',c='gold',edgecolors='k',s=60)  # 画散点图, alpha=0.6 表示不透明度为 0.6




plt.xlim(0.8,5.5)
plt.ylim(5,115)

#plt.legend(['Pressure','Melting degree','Temperature'], loc='lower center',fontsize=10, ncol=1)


plt.legend(['Melting degree','Temperature','Pressure'], loc='lower right',fontsize=8, ncol=1)

plt.plot([0,10],[90,90],'b--',alpha=0.3,linewidth=0.5)



plt.text(2.4,10,'Peridotite:313',fontsize=10)
plt.text(0.9,100,'(a)',fontsize=12)


#-----------------------------------------------------------------------------------------transitional
plt.subplot(3,1,2)
#ax2 = fig.add_subplot(132)
#ax2.set_position([0.1,0.2,0.8,0.79])
#
#locate=[1,2,3,4,5]




#s=[p0_total,p1_total,p2_total,p3_total,p4_total]
plt.scatter([1, 2, 3, 4, 5],[m0_ratio_t,m1_ratio_t,m2_ratio_t,m3_ratio_t,m4_ratio_t],marker='s',c='orangered',edgecolors='k',s=50)  # 画散点图, alpha=0.6 表示不透明度为 0.6

plt.scatter([1, 2, 3, 4, 5],[t0_ratio_t,t1_ratio_t,t2_ratio_t,t3_ratio_t,t4_ratio_t],c='g',edgecolors='k',s=50)  # 画散点图, alpha=0.6 表示不透明度为 0.6

ln1=plt.scatter([1, 2, 3, 4, 5],[p0_ratio_t,p1_ratio_t,p2_ratio_t,p3_ratio_t,p4_ratio_t],marker='^',c='gold',edgecolors='k',s=60)  # 画散点图, alpha=0.6 表示不透明度为 0.6




plt.xlim(0.8,5.5)
plt.ylim(5,115)

#plt.legend(['Pressure','Melting degree','Temperature'], loc='lower center',fontsize=10, ncol=1)
plt.plot([0,10],[90,90],'b--',alpha=0.3,linewidth=0.5)

plt.ylabel('Accuracy(%)',fontsize=12)


plt.text(2.4,8,'Transitional:103',fontsize=10)
plt.text(0.9,100,'(b)',fontsize=12)


#---------------------------------------------------------------------------------------mafic
#########----------------------------------
plt.subplot(3,1,3)

#ax2 = fig.add_subplot(133)
#ax2.set_position([0.1,0.2,0.8,0.79])
#locate=[1,2,3,4,5]



#s=[p0_total,p1_total,p2_total,p3_total,p4_total]

plt.scatter([1, 2, 3, 4, 5],[m0_ratio_m,m1_ratio_m,m2_ratio_m,m3_ratio_m,m4_ratio_m],marker='s',c='orangered',edgecolors='k',s=50)  # 画散点图, alpha=0.6 表示不透明度为 0.6

plt.scatter([1, 2, 3, 4, 5],[t0_ratio_m,t1_ratio_m,t2_ratio_m,t3_ratio_m,t4_ratio_m],c='g',edgecolors='k',s=50)  # 画散点图, alpha=0.6 表示不透明度为 0.6

ln1=plt.scatter([1, 2, 3, 4, 5],[p0_ratio_m,p1_ratio_m,p2_ratio_m,p3_ratio_m,p4_ratio_m],marker='^',c='gold',edgecolors='k',s=60)  # 画散点图, alpha=0.6 表示不透明度为 0.6


plt.xlim(0.8,5.5)


plt.ylim(5,115)



# #plt.text(a,z,'Sigmoid(%s)'%(a),fontdict=font)
# 
#d=p0_total
#plt.text(1+0.1,p0_ratio,'%d'%(d),fontsize=10)
# 
#d=p1_total
#plt.text(2+0.1,p1_ratio-3,'%d'%(d),fontsize=10)
# 
#d=p2_total
#plt.text(3+0.1,p2_ratio,'%d'%(d),fontsize=10)
## 
#d=p3_total
#plt.text(4+0.1,p3_ratio,'%d'%(d),fontsize=10)
## 
#d=p4_total
#plt.text(5+0.1,p4_ratio,'%d'%(d),fontsize=10)
# 
# 
# 
# #-----------------meltingdegree
#d=m0_total
#plt.text(1+0.1,m0_ratio,'%d'%(d),fontsize=10)
## 
#d=m1_total
#plt.text(2+0.1,m1_ratio,'%d'%(d) ,fontsize=10)
## 
#d=m2_total
#plt.text(3+0.1,m2_ratio-2,'%d'%(d) ,fontsize=10)
## 
#d=m3_total
#plt.text(4+0.1,m3_ratio,'%d'%(d) ,fontsize=10)
## 
#d=m4_total
#plt.text(5+0.1,m4_ratio,'%d'%(d),fontsize=10)
## 


#-----------------Temperature

#d=t0_total
#plt.text(1+0.1,t0_ratio,'%d'%(d),fontsize=10)
## 
#d=t1_total
#plt.text(2+0.1,t1_ratio,'%d'%(d) ,fontsize=10)
## 
#d=t2_total
#plt.text(3+0.1,t2_ratio,'%d'%(d) ,fontsize=10)
## 
#d=t3_total
#plt.text(4+0.1,t3_ratio_m,'%d'%(d) ,fontsize=10)
## 
#d=t4_total
#plt.text(5+0.1,t4_ratio_m-3,'%d'%(d),fontsize=10)


plt.xlabel('Pressure (GPa); Melting degree; Temperature (℃)',fontsize=12)

plt.plot([0,10],[90,90],'b--',alpha=0.3,linewidth=0.5)


xlab3 = [ "P(GPa): 0-2", "2-4", "4-6","6-8","8+"]
xlab1 = ["MD: 0-0.2", "0.2-0.4", "0.4-0.6","0.6-0.8","0.8-1"] 
xlab2 = ["T(℃): <1200", "1200-1400", "1400-1600","1600-1800","1800+"] 

xlabels = [f"{x1}\n{x2}\n{x3}" for x1, x2,x3 in zip(xlab1,xlab2,xlab3)]

plt.text(2.8,10,'Mafic:499',fontsize=10)
plt.text(0.9,100,'(c)',fontsize=12)



plt.xticks([1, 2, 3, 4, 5],xlabels  )

#plt.ylim([])


#----save fig
plt.savefig('accuracy_vs_meltingdegree_checkdata',dpi=300)
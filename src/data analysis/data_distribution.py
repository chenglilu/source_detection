#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 16:00:11 2021

@author: lilucheng
"""

#show the data distribution
#compare the distribution of three groups





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



#chose 

group1=np.where(y==1)
group10=group1[0]
data_g1=X[group10]
label_g1=y[group10]

group2=np.where(y==2)
group20=group2[0]
data_g2=X[group20]
label_g2=y[group20]


group3=np.where(y==3)
group30=group3[0]
data_g3=X[group30]
label_g3=y[group30]


#-------------------------------------------------------
#prepare mgo,al2o3,

sio2=np.zeros((915,1))
for i in range(0,915):   
   sio2[i] = X[i,0] 


tio2=np.zeros((915,1))
for i in range(0,915):   
   tio2[i] = X[i,1] 

al2o3=np.zeros((915,1))
for i in range(0,915):   
   al2o3[i] = X[i,2] 
   
   
cr2o3=np.zeros((915,1))
for i in range(0,915):   
   cr2o3[i] = X[i,3] 
   
feo=np.zeros((915,1))
for i in range(0,915):   
   feo[i] = X[i,4]    

mno=np.zeros((915,1))
for i in range(0,915):   
   mno[i] = X[i,5] 
   
mgo=np.zeros((915,1))
for i in range(0,915):   
   mgo[i] = X[i,6] 


cao=np.zeros((915,1))
for i in range(0,915):   
   cao[i] = X[i,7] 

na2o=np.zeros((915,1))
for i in range(0,915):   
   na2o[i] = X[i,8] 
   
   
k2o=np.zeros((915,1))
for i in range(0,915):   
   k2o[i] = X[i,9]    
   

mgjh=np.zeros((915,1))
for i in range(0,915):   
   mgjh[i] = data.iloc[i+1,19]  
   
#-------------------------------------------------------
#start to plot
sio2_g1 = sio2[group10]
sio2_g2 = sio2[group20]
sio2_g3 = sio2[group30]

 
tio2_g1 = tio2[group10]
tio2_g2 = tio2[group20]
tio2_g3 = tio2[group30]
 

al2o3_g1 = al2o3[group10]
al2o3_g2 = al2o3[group20]
al2o3_g3 = al2o3[group30]
 
 
cr2o3_g1 = cr2o3[group10]
cr2o3_g2 = cr2o3[group20]
cr2o3_g3 = cr2o3[group30]
   

feo_g1 = feo[group10]    
feo_g2 = feo[group20]    
feo_g3 = feo[group30]    



mno_g1 = mno[group10]
mno_g2 = mno[group20]
mno_g3 = mno[group30]

   
mgo_g1 = mgo[group10]
mgo_g2 = mgo[group20]
mgo_g3 = mgo[group30]


cao_g1 = cao[group10]
cao_g2 = cao[group20]
cao_g3 = cao[group30]

 
na2o_g1 = na2o[group10]
na2o_g2 = na2o[group20]
na2o_g3 = na2o[group30]
   
   
 
k2o_g1 = k2o[group10]
k2o_g2 = k2o[group20]
k2o_g3 = k2o[group30]
   
   
   



#---------------------------------group them using already known labels
   
   

plt.figure()


plt.subplots_adjust(left=0.1, right=0.95,bottom=0.12,top=0.95, wspace=0.45, hspace=0.55)
plt1=plt.subplot(3,3,1)

n, bins, patches = plt.hist(sio2_g1, 20,density=False, facecolor='limegreen',edgecolor='limegreen', alpha=1,histtype='step')
n, bins, patches = plt.hist(sio2_g2,20, density=False,facecolor='deepskyblue', edgecolor='r', alpha=1,histtype='step')
n, bins, patches = plt.hist(sio2_g3,20, density=False, facecolor='0.5',edgecolor='k', alpha=1,histtype='step')



plt.xlabel('Si$\mathregular{O_2}$(wt%)',fontsize=10) 
#plt.ylabel('Probability')
plt.ylabel('Number')

plt.figtext(0.04,0.97,'(a)',color='black',fontsize=11)


#-----
plt1=plt.subplot(3,3,2)

n, bins, patches = plt.hist(tio2_g1,20, density=False, edgecolor='limegreen', alpha=1,histtype='step')
n, bins, patches = plt.hist(tio2_g2, 20,density=False, edgecolor='r', alpha=1,histtype='step')
n, bins, patches = plt.hist(tio2_g3,20, density=False, edgecolor='k', alpha=1,histtype='step')

plt.legend(['Peridotitic','Transitional','Mafic'], loc='upper right',fontsize=6)


plt.xlabel('Ti$\mathregular{O_2}$(wt%)',fontsize=10) 
#plt.ylabel('Probability')
plt.ylabel('Number')
plt.figtext(0.35,0.97,'(b)',color='black',fontsize=11)


#-----
plt1=plt.subplot(3,3,3)

n, bins, patches = plt.hist(al2o3_g1, 20,density=False, edgecolor='limegreen', alpha=1,histtype='step')
n, bins, patches = plt.hist(al2o3_g2, 20,density=False, edgecolor='r', alpha=1,histtype='step')
n, bins, patches = plt.hist(al2o3_g3, 20,density=False, edgecolor='k', alpha=1,histtype='step')



plt.xlabel('A$\mathregular{l_2}$$\mathregular{O_3}$(wt%)',fontsize=11) 
#plt.ylabel('Probability')
plt.ylabel('Number')
plt.figtext(0.68,0.97,'(c)',color='black',fontsize=11)


#-----
plt1=plt.subplot(3,3,4)

n, bins, patches = plt.hist(mgo_g1, 20,density=False, edgecolor='limegreen', alpha=1,histtype='step')
n, bins, patches = plt.hist(mgo_g2, 20,density=False, edgecolor='r', alpha=1,histtype='step')
n, bins, patches = plt.hist(mgo_g3, 20,density=False, edgecolor='k', alpha=1,histtype='step')



plt.xlabel('MgO(wt%)',fontsize=10) 
#plt.ylabel('Probability')
plt.ylabel('Number')

plt.figtext(0.04,0.65,'(d)',color='black',fontsize=11)


#-----
plt1=plt.subplot(3,3,5)

n, bins, patches = plt.hist(feo_g1,20, density=False, edgecolor='limegreen', alpha=1,histtype='step')
n, bins, patches = plt.hist(feo_g2,20, density=False, edgecolor='r', alpha=1,histtype='step')
n, bins, patches = plt.hist(feo_g3, 20,density=False, edgecolor='k', alpha=1,histtype='step')



plt.xlabel('FeO(wt%)',fontsize=10) 
#plt.ylabel('Probability')
plt.ylabel('Number')
plt.figtext(0.35,0.65,'(e)',color='black',fontsize=11)



#-----
plt1=plt.subplot(3,3,6)

n, bins, patches = plt.hist(mno_g1,20, density=False, edgecolor='limegreen', alpha=1,histtype='step')
n, bins, patches = plt.hist(mno_g2,20, density=False, edgecolor='r', alpha=1,histtype='step')
n, bins, patches = plt.hist(mno_g3,20, density=False, edgecolor='k', alpha=1,histtype='step')



plt.xlabel('MnO(wt%)',fontsize=10) 
#plt.ylabel('Probability')
plt.ylabel('Number')
plt.figtext(0.68,0.65,'(f)',color='black',fontsize=11)


#-----
plt1=plt.subplot(3,3,7)

n, bins, patches = plt.hist(cao_g1, 20,density=False, edgecolor='limegreen', alpha=1,histtype='step')
n, bins, patches = plt.hist(cao_g2,20, density=False, edgecolor='r', alpha=1,histtype='step')
n, bins, patches = plt.hist(cao_g3,20, density=False, edgecolor='k', alpha=1,histtype='step')



plt.xlabel('CaO(wt%)',fontsize=10) 
#plt.ylabel('Probability')
plt.ylabel('Number')
plt.figtext(0.04,0.34,'(g)',color='black',fontsize=11)




#-----
plt1=plt.subplot(3,3,8)

n, bins, patches = plt.hist(na2o_g1,20, density=False, edgecolor='limegreen', alpha=1,histtype='step')
n, bins, patches = plt.hist(na2o_g2,20, density=False, edgecolor='r', alpha=1,histtype='step')
n, bins, patches = plt.hist(na2o_g3, 20,density=False, edgecolor='k', alpha=1,histtype='step')



plt.xlabel('N$\mathregular{a_2}$O(wt%)',fontsize=10) 
#plt.ylabel('Probability')
plt.ylabel('Number')
plt.figtext(0.35,0.34,'(h)',color='black',fontsize=11)


#-----
plt1=plt.subplot(3,3,9)

n, bins, patches = plt.hist(k2o_g1,20, density=False, edgecolor='limegreen', alpha=1,histtype='step')
n, bins, patches = plt.hist(k2o_g2,20, density=False, edgecolor='r', alpha=1,histtype='step')
n, bins, patches = plt.hist(k2o_g3,20, density=False, edgecolor='k', alpha=1,histtype='step')


plt.xlabel('$\mathregular{K_2}$O(wt%)',fontsize=10) 
#plt.ylabel('Probability')
plt.ylabel('Number')

plt.figtext(0.68,0.34,'(i)',color='black',fontsize=11)



plt.savefig('data_prepare_distribution',dpi=300)


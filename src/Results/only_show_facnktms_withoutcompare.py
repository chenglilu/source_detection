#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 20:46:17 2021

@author: lilucheng
"""




import pandas as pd
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from numpy import *


#-------------------------------------------------------
# Read the file  dataframe
data = pd.read_excel('data2.xlsx',header = None,skipfooter= 1,index_col=1)

#change into the data we need float
#train data determined from dataframe
Traindata = np.zeros((928,18))  
for i in range(0,928):  
 for j in range(0,18):  
   Traindata[i][j] = data.iloc[i+1,j+6]  


#change nan into 0
for i in range(0,928):  
 for j in range(0,18):  
  if (np.isnan(Traindata[i][j])):
      Traindata[i][j]= 0
  
   

#lable from dataframe
Group=np.zeros((928,1))
for i in range(0,928):   
   Group[i] = data.iloc[i+1,24]  


#-------------------------------------------------------
X = Traindata
y = Group
#class_names={'Mafic','Transitional','Peridotite'}



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state = 0)


clf = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 10), random_state=1)





clf = clf.fit(X_train, y_train)



#-------------------------------------------------------
   
y_result=clf.predict(X)

y_result=y_result.reshape(-1,1)


difference_total=np.zeros((763,1))
difference_total=y-y_result


idx=np.where(difference_total==0)  
idx0=idx[0]
Same_rate=len(idx0)/len(y_result)
print('Same number of all the data',Same_rate)   



#-----peridotite right and wrong
idx1r=np.where((y==1) & (difference_total==0))
#idx1w=np.where((y==1) & (difference_total!=0))

idx1r0=idx1r[0]
#idx1w0=idx1w[0]

#-----peridotite right and wrong

idx2r=np.where((y==2) & (difference_total==0))
#idx2w=np.where((y==2) & (difference_total!=0))
idx2r0=idx2r[0]
#idx2w0=idx2w[0]

#-----peridotite right and wrong


idx3r=np.where((y==3) & (difference_total==0))
#idx3w=np.where((y==3) & (difference_total!=0))

idx3r0=idx3r[0]
#idx3w0=idx3w[0]

#-------------------------------------------------------
# mgo,mg#,cati,sica

   
mgo=np.zeros((928,1))
for i in range(0,928):   
   mgo[i] = X[i,6] 

mg=np.zeros((928,1))
for i in range(0,928):   
   mg[i] = X[i,13] 
   
cati=np.zeros((928,1))
for i in range(0,928):   
   cati[i] = X[i,16] 

sica=np.zeros((928,1))
for i in range(0,928):   
   sica[i] = X[i,17]   
   
fcms=np.zeros((928,1))
for i in range(0,928):   
   fcms[i] = X[i,15]    
 

#-------------------------------------------------------
###############################
fig=plt.figure()


plt.subplots_adjust(left=0.17, right=0.95,bottom=0.12,top=0.95, wspace=0.4, hspace=0.35)


ax1=plt.subplot(2,2,1)


xx=[0,1]
yy1=[0.37,0.37]
yy2=[0.05,0.05]
plt.plot(xx,yy1,'b--',alpha=0.3,linewidth=0.5)
plt.plot(xx,yy2,'b--',alpha=0.3,linewidth=0.5)



l1=plt.scatter(mg[idx3r0],fcms[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
l2=plt.scatter(mg[idx2r0],fcms[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
l3=plt.scatter(mg[idx1r0],fcms[idx1r0],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=15,linewidth=0.5)



# =============================================================================
# l4=plt.scatter(mg[idx3w0],fcms[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)
# l5=plt.scatter(mg[idx2w0],fcms[idx2w0],marker='s',edgecolors='orangered',facecolor='gold',s=15,linewidth=0.8)
# l6=plt.scatter(mg[idx1w0],fcms[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)
# 
# =============================================================================

#ax = fig.add_subplot(111)



plt.xlabel('Mg#',fontsize=11)          
plt.ylabel('FCKANTMS',fontsize=11) 

plt.xlim(xx)

plt.figtext(0.45,0.9,'(a)',color='black',fontsize=12)


plt.legend([l1,l2,l3,l4,l5,l6],['Mafic', 'Transitional', 'Peridotite'], loc='lower left',fontsize=6)

#-------------------------------------
ax2=plt.subplot(2,2,2)

xx=[0,45]
yy1=[0.37,0.37]
yy2=[0.05,0.05]
plt.plot(xx,yy1,'b--',alpha=0.3,linewidth=0.5)
plt.plot(xx,yy2,'b--',alpha=0.3,linewidth=0.5)



plt.scatter(mgo[idx3r0],fcms[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(mgo[idx2r0],fcms[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
plt.scatter(mgo[idx1r0],fcms[idx1r0],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=15,linewidth=0.5)



# =============================================================================
# plt.scatter(mgo[idx3w0],fcms[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)
# plt.scatter(mgo[idx2w0],fcms[idx2w0],marker='s',edgecolors='orangered',facecolor='gold',s=15,linewidth=0.8)
# plt.scatter(mgo[idx1w0],fcms[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)
# 
# 
# =============================================================================



plt.xlim(xx)

plt.xlabel('MgO',fontsize=11)          
plt.ylabel('FCKANTMS',fontsize=11) 
plt.figtext(0.9,0.9,'(b)',color='black',fontsize=12)


#-------------------------------------
ax3=plt.subplot(2,2,3)


xx=[0,3]
yy1=[0.37,0.37]
yy2=[0.05,0.05]
plt.plot(xx,yy1,'b--',alpha=0.3,linewidth=0.5)
plt.plot(xx,yy2,'b--',alpha=0.3,linewidth=0.5)



plt.scatter(sica[idx3r0],fcms[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(sica[idx2r0],fcms[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
plt.scatter(sica[idx1r0],fcms[idx1r0],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=15,linewidth=0.5)



# =============================================================================
# plt.scatter(sica[idx3w0],fcms[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)
# plt.scatter(sica[idx2w0],fcms[idx2w0],marker='s',edgecolors='orangered',facecolor='gold',s=15,linewidth=0.8)
# plt.scatter(sica[idx1w0],fcms[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)
# 
# 
# =============================================================================




plt.xlim(xx)

plt.xlabel('ln(Si$\mathregular{O_2}$/CaO+$\mathregular{Na_2}$O+Ti$\mathregular{O_2}$)',fontsize=12)   

plt.ylabel('FCKANTMS',fontsize=11) 

plt.figtext(0.45,0.42,'(c)',color='black',fontsize=12)


#-------------------------------------
ax4=plt.subplot(2,2,4)

xx=[-2,6]
yy1=[0.37,0.37]
yy2=[0.05,0.05]
plt.plot(xx,yy1,'b--',alpha=0.3,linewidth=0.5)
plt.plot(xx,yy2,'b--',alpha=0.3,linewidth=0.5)



plt.scatter(cati[idx3r0],fcms[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(cati[idx2r0],fcms[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
plt.scatter(cati[idx1r0],fcms[idx1r0],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=15,linewidth=0.5)



# =============================================================================
# plt.scatter(cati[idx3w0],fcms[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)
# plt.scatter(cati[idx2w0],fcms[idx2w0],marker='s',edgecolors='orangered',facecolor='gold',s=15,linewidth=0.8)
# plt.scatter(cati[idx1w0],fcms[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)
# 
# 
# =============================================================================





plt.figtext(0.64,0.38,'Mafic',color='black',fontsize=7)
plt.figtext(0.64,0.318,'M & T',color='black',fontsize=7)
plt.figtext(0.64,0.252,'Mafic,',color='black',fontsize=7)
plt.figtext(0.639,0.22,'Transitional',color='black',fontsize=7)
plt.figtext(0.64,0.19,'and Peridotite',color='black',fontsize=7)




plt.xlabel('ln(CaO/Ti$\mathregular{O_2}$)',fontsize=12)          
plt.xlim(xx)


plt.figtext(0.9,0.42,'(d)',color='black',fontsize=12)


plt.ylabel('FCKANTMS',fontsize=11) 

plt.savefig('only_melt_result_withoutcompare_fackantms_fcms',dpi=300)

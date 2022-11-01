#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 10:35:55 2021

@author: lilucheng
"""

#compare ann result to 




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
#prepare mgo,cao


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


fc3ms=np.zeros((915,1))
for i in range(0,915):   
   fc3ms[i] = data.iloc[i+1,18]  
#-------------------------------------------------------
#start to plot


plt.figure()



plt.subplots_adjust(left=0.1, right=0.95,bottom=0.22,top=0.9, wspace=0.3, hspace=0.2)
ax1=plt.subplot(1,2,1)



plt.scatter(mgo[idx3r0],cao[idx3r0],marker='^',c='',edgecolors='0.5',s=30,linewidth=0.5)
plt.scatter(mgo[idx2r0],cao[idx2r0],marker='s',c='',edgecolors='k',s=30,linewidth=0.5)
plt.scatter(mgo[idx1r0],cao[idx1r0],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=30,linewidth=0.5)



plt.scatter(mgo[idx3w0],cao[idx3w0],marker='^',c='',edgecolors='orangered',s=30,linewidth=0.8)
plt.scatter(mgo[idx2w0],cao[idx2w0],marker='s',edgecolors='orangered',facecolor='gold',s=30,linewidth=0.8)
plt.scatter(mgo[idx1w0],cao[idx1w0],marker='x',c='orangered',s=30,linewidth=0.8)

#cao=13.81-0.274MgO  from Herzberg & Asimow 2008

#rect = plt.Rectangle((0.65,42),0.2,15,fill='False',facecolor='none',edgecolor='k')
#plt1.add_patch(rect)
plt.xlabel('MgO(wt%)',fontsize=12)          
plt.ylabel('CaO(wt%)',fontsize=12) 

#plt.figtext(0.28,0.9,'(a)',color='black',fontsize=12)

#plt.legend(['Mafic R', 'Transitional R', 'Peridotite R','Mafic W','Transitional W','Peridotite W'], loc='top right',fontsize=8)
x1=[0,45]
y1=[13.81,1.48]

plt.xlim(0,45)
plt.text(27,7,'Peridotite \npartial melts')
plt.text(22,0.7,'Pyroxenite \npartial melts')


plt.plot(x1,y1,color='b')

plt.text(0,25.5,'(a)',fontsize=14)

#-------------------------------compare to FC3MS

ax2=plt.subplot(1,2,2)


l1=plt.scatter(mgo[idx3r0],fc3ms[idx3r0],marker='^',c='',edgecolors='0.5',s=30,linewidth=0.5)
l2=plt.scatter(mgo[idx2r0],fc3ms[idx2r0],marker='s',c='',edgecolors='k',s=30,linewidth=0.5)
l3=plt.scatter(mgo[idx1r0],fc3ms[idx1r0],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=30,linewidth=0.5)

plt.scatter(mgo[idx3r0],fc3ms[idx3r0],marker='^',c='',edgecolors='0.5',s=30,linewidth=0.5)


l4=plt.scatter(mgo[idx3w0],fc3ms[idx3w0],marker='^',c='',edgecolors='orangered',s=30,linewidth=0.8)
l5=plt.scatter(mgo[idx2w0],fc3ms[idx2w0],marker='s',edgecolors='orangered',facecolor='gold',s=30,linewidth=0.8)
l6=plt.scatter(mgo[idx1w0],fc3ms[idx1w0],marker='x',c='orangered',s=30,linewidth=0.8)



#cao=13.81-0.274MgO  from Herzberg & Asimow 2008

#rect = plt.Rectangle((0.65,42),0.2,15,fill='False',facecolor='none',edgecolor='k')
#plt1.add_patch(rect)
plt.xlabel('MgO(wt%)',fontsize=12)          
plt.ylabel('FC3MS',fontsize=12) 

#plt.figtext(0.28,0.9,'(a)',color='black',fontsize=12)

plt.legend([l1,l2,l3,l4,l5,l6],['Mafic R', 'Transitional R', 'Peridotite R','Mafic W','Transitional W','Peridotite W'], loc='top right',fontsize=7.5)
x1=[0,45]

yy2=[0.65,0.65]
plt.plot(x1,yy2,'b--',alpha=0.9,linewidth=1)

plt.text(23,0.85,'Pyroxenite melt')
plt.text(24,0.15,'Peridotite melt')



plt.xlim(0,45)

#plt.ylim(-3.2,4)


plt.text(0.5,4.2,'(b)',fontsize=14)


#----save fig
plt.savefig('compare_caomgo_fc3ms_checkdata',dpi=300)

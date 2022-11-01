#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 13:28:17 2021

@author: lilucheng
"""

#plot results and discusses
#3 different groups with right and wrong =total 6 labels
# mafic group right and wrong
# transitional group right and wrong
#peridotite group right and wrong


#ANN  multiple layers



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
Traindata = np.zeros((928,10))  
for i in range(0,928):  
 for j in range(0,10):  
   Traindata[i][j] = data.iloc[i+1,j+6]  


#change nan into 0
for i in range(0,928):  
 for j in range(0,10):  
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


difference_total=np.zeros((928,1))
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

#-----peridotite right and wrong

idx2r=np.where((y==2) & (difference_total==0))
idx2w=np.where((y==2) & (difference_total!=0))
idx2r0=idx2r[0]
idx2w0=idx2w[0]

#-----peridotite right and wrong


idx3r=np.where((y==3) & (difference_total==0))
idx3w=np.where((y==3) & (difference_total!=0))

idx3r0=idx3r[0]
idx3w0=idx3w[0]

#-------------------------------------------------------
#prepare mgo,al2o3,

sio2=np.zeros((928,1))
for i in range(0,928):   
   sio2[i] = X[i,0] 


tio2=np.zeros((928,1))
for i in range(0,928):   
   tio2[i] = X[i,1] 

al2o3=np.zeros((928,1))
for i in range(0,928):   
   al2o3[i] = X[i,2] 
   
   
cr2o3=np.zeros((928,1))
for i in range(0,928):   
   cr2o3[i] = X[i,3] 
   
feo=np.zeros((928,1))
for i in range(0,928):   
   feo[i] = X[i,4]    

mno=np.zeros((928,1))
for i in range(0,928):   
   mno[i] = X[i,5] 
   
mgo=np.zeros((928,1))
for i in range(0,928):   
   mgo[i] = X[i,6] 


cao=np.zeros((928,1))
for i in range(0,928):   
   cao[i] = X[i,7] 

na2o=np.zeros((928,1))
for i in range(0,928):   
   na2o[i] = X[i,8] 
   
   
k2o=np.zeros((928,1))
for i in range(0,928):   
   k2o[i] = X[i,9]    

#-------------------------------------------------------
#start to plot


plt.figure()


plt.subplots_adjust(left=0.1, right=0.95,bottom=0.12,top=0.95, wspace=0.42, hspace=0)
plt.subplot(3,3,1)


plt.scatter(mgo[idx3r0],sio2[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(mgo[idx2r0],sio2[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
plt.scatter(mgo[idx1r0],sio2[idx1r0],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=15,linewidth=0.5)



plt.scatter(mgo[idx3w0],sio2[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)
plt.scatter(mgo[idx2w0],sio2[idx2w0],marker='s',edgecolors='orangered',facecolor='gold',s=15,linewidth=0.8)
plt.scatter(mgo[idx1w0],sio2[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)



#plt.xlabel('MgO(%)',fontsize=12)          
plt.ylabel('Si$\mathregular{O_2}$(%)',fontsize=11) 

plt.figtext(0.28,0.9,'(a)',color='black',fontsize=12)

#-----
plt2=plt.subplot(3,3,2)

plt.scatter(mgo[idx3r0],tio2[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(mgo[idx2r0],tio2[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
plt.scatter(mgo[idx1r0],tio2[idx1r0],marker='x',c='deepskyblue',s=15,linewidth=0.5)



plt.scatter(mgo[idx3w0],tio2[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)
plt.scatter(mgo[idx2w0],tio2[idx2w0],marker='s',edgecolors='orangered',facecolor='gold',s=15,linewidth=0.8)
plt.scatter(mgo[idx1w0],tio2[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)






#plt.xlabel('MgO(%)',fontsize=12)          
plt.ylabel('Ti$\mathregular{O_2}$(%)',fontsize=11) 

plt.figtext(0.6,0.9,'(b)',color='black',fontsize=12)

plt2.legend(['Mafic R', 'Transitional R', 'Peridotite R','Mafic W','Transitional W','Peridotite W'], loc='center right',fontsize=4)
#plt.figtext(0.55,0.85,'S_ID*RA=74%',color='black',fontsize=12)


#-----
plt.subplot(3,3,3)

plt.scatter(mgo[idx3r0],al2o3[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(mgo[idx2r0],al2o3[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
plt.scatter(mgo[idx1r0],al2o3[idx1r0],marker='x',c='deepskyblue',s=15,linewidth=0.5)




plt.scatter(mgo[idx3w0],al2o3[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)
plt.scatter(mgo[idx2w0],al2o3[idx2w0],marker='s',edgecolors='orangered',facecolor='y',s=15,linewidth=0.8)
plt.scatter(mgo[idx1w0],al2o3[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)





#plt.xlabel('MgO(%)',fontsize=11)          
plt.ylabel('$\mathregular{Al_2}$$\mathregular{O_3}$(%)',fontsize=11) 

plt.figtext(0.91,0.9,'(c)',color='black',fontsize=12)

#-----
plt.subplot(3,3,4)

plt.scatter(mgo[idx2r0],cr2o3[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
plt.scatter(mgo[idx3r0],cr2o3[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(mgo[idx1r0],cr2o3[idx1r0],marker='x',c='deepskyblue',s=15,linewidth=0.5)


plt.scatter(mgo[idx2w0],cr2o3[idx2w0],marker='s',edgecolors='orangered',facecolor='y',s=15,linewidth=0.8)

plt.scatter(mgo[idx3w0],cr2o3[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)

plt.scatter(mgo[idx1w0],cr2o3[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)



#plt.xlabel('MgO(%)',fontsize=12)          
plt.ylabel('$\mathregular{Cr_2}$$\mathregular{O_3}$(%)',fontsize=11) 

plt.figtext(0.28,0.61,'(d)',color='black',fontsize=12)

#-----
plt.subplot(3,3,5)

plt.scatter(mgo[idx3r0],feo[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(mgo[idx2r0],feo[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
plt.scatter(mgo[idx1r0],feo[idx1r0],marker='x',c='deepskyblue',s=15,linewidth=0.5)


plt.scatter(mgo[idx3w0],feo[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)
plt.scatter(mgo[idx2w0],feo[idx2w0],marker='s',edgecolors='orangered',facecolor='y',s=15,linewidth=0.8)
plt.scatter(mgo[idx1w0],feo[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)




#plt.xlabel('MgO(%)',fontsize=12)          
plt.ylabel('FeO(%)',fontsize=11) 

plt.figtext(0.6,0.61,'(e)',color='black',fontsize=12)

#-----
plt.subplot(3,3,6)

plt.scatter(mgo[idx3r0],mno[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(mgo[idx2r0],mno[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
plt.scatter(mgo[idx1r0],mno[idx1r0],marker='x',c='deepskyblue',s=15,linewidth=0.5)


plt.scatter(mgo[idx3w0],mno[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)
plt.scatter(mgo[idx2w0],mno[idx2w0],marker='s',edgecolors='orangered',facecolor='y',s=15,linewidth=0.8)
plt.scatter(mgo[idx1w0],mno[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)




#plt.xlabel('MgO(%)',fontsize=12)          
plt.ylabel('MnO(%)',fontsize=11) 

plt.figtext(0.91,0.61,'(f)',color='black',fontsize=12)


#-----
plt.subplot(3,3,7)

plt.scatter(mgo[idx3r0],cao[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(mgo[idx2r0],cao[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
plt.scatter(mgo[idx1r0],cao[idx1r0],marker='x',c='deepskyblue',s=15,linewidth=0.5)

plt.scatter(mgo[idx3w0],cao[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)
plt.scatter(mgo[idx2w0],cao[idx2w0],marker='s',edgecolors='orangered',facecolor='y',s=15,linewidth=0.8)
plt.scatter(mgo[idx1w0],cao[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)







plt.xlabel('MgO(%)',fontsize=11)          
plt.ylabel('CaO(%)',fontsize=11) 

plt.figtext(0.28,0.34,'(g)',color='black',fontsize=12)


#-----
plt.subplot(3,3,8)

plt.scatter(mgo[idx3r0],na2o[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)

plt.scatter(mgo[idx2r0],na2o[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)

plt.scatter(mgo[idx1r0],na2o[idx1r0],marker='x',c='deepskyblue',s=15,linewidth=0.5)





plt.scatter(mgo[idx3w0],na2o[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)

plt.scatter(mgo[idx2w0],na2o[idx2w0],marker='s',edgecolors='orangered',facecolor='y',s=15,linewidth=0.8)

plt.scatter(mgo[idx1w0],na2o[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)






plt.xlabel('MgO(%)',fontsize=11)          
plt.ylabel('$\mathregular{Na_2}$O(%)',fontsize=11) 

plt.figtext(0.595,0.34,'(h)',color='black',fontsize=12)


#-----
plt.subplot(3,3,9)


plt.scatter(mgo[idx3r0],k2o[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(mgo[idx2r0],k2o[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)

plt.scatter(mgo[idx1r0],k2o[idx1r0],marker='x',c='deepskyblue',s=15,linewidth=0.5)



plt.scatter(mgo[idx3w0],k2o[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)

plt.scatter(mgo[idx2w0],k2o[idx2w0],marker='s',edgecolors='orangered',facecolor='y',s=15,linewidth=0.8)

plt.scatter(mgo[idx1w0],k2o[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)




plt.xlabel('MgO(%)',fontsize=11)          
plt.ylabel('$\mathregular{K_2}$O(%)',fontsize=11) 

plt.figtext(0.91,0.34,'(i)',color='black',fontsize=12)

#----save fig
plt.savefig('resultcompare',dpi=300)

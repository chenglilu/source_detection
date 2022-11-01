#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 16:02:17 2021

@author: lilucheng
"""

#determine the problem of transitional


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



#D=X
#idq=np.where((D[:,0]<30) | (D[:,0]>65))  
#idq0=idq[0]
#newX=np.delete(D,idq0,0)

#newy=np.delete(y,idq0)
#newy=newy.reshape(-1,1)

newX=X
newy=y

X_train, X_test, y_train, y_test = train_test_split(newX, newy, train_size=0.9, random_state = 0)


clf = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 10), random_state=1)


clf = clf.fit(X_train, y_train)




accuracy_ANN = clf.score(X_test, y_test)

print('Accuracy Neural network test:', accuracy_ANN)


#visilize
# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
accuracy_ANNtrain = clf.score(X_train, y_train)

print('Accuracy Neural network of train:', accuracy_ANNtrain)



y_result=clf.predict(newX) 
y_result=y_result.reshape(-1,1) 
difference_total=np.zeros((928,1))
difference_total=newy-y_result
idx=np.where(difference_total==0)  
idx0=idx[0]
Same_alldata=len(idx0)/len(y_result)
print('Accuracy Neural network of all data:', Same_alldata)



plt.show()


#------------------------------------------------------------------------------------------
#read data from transitio


datat = pd.read_excel('data_tran.xlsx',header = None,skipfooter= 1,index_col=1)



data_tran = np.zeros((200,10))  
for i in range(0,200):  
 for j in range(0,10):  
   data_tran[i][j] = datat.iloc[i+1,j+1]  
   
data_tran_result=clf.predict(data_tran)

data_tran_result=data_tran_result.reshape(-1,1)



real_tran=np.zeros((200,1)) 
for i in range(0,200):   
   real_tran[i] = datat.iloc[i+1,11]  

difference_tran=np.zeros((200,1))
difference_tran=data_tran_result-real_tran
idx=np.where(difference_tran==0)  
idx0=idx[0]
Same_tran=len(idx0)/len(real_tran)
print('Accuracy Neural network of transitional:', Same_tran)



fig, ax = plt.subplots()

plot_confusion_matrix(clf, data_tran, real_tran,normalize='true',ax=ax)  



fig.savefig('data_transitional',dpi=300)




#------------------------------------------------------------------------------------------
#read data from random data 2  which include mafic and transitional


datar = pd.read_excel('randomdata2.xlsx',header = None,skipfooter= 1,index_col=1)



data_random = np.zeros((1000,10))  
for i in range(0,1000):  
 for j in range(0,10):  
   data_random[i][j] = datar.iloc[i+1,j+1]  
   
data_random_result=clf.predict(data_random)

data_random_result=data_random_result.reshape(-1,1)



real_random=np.zeros((1000,1)) 
for i in range(0,1000):   
   real_random[i] = datar.iloc[i+1,46]  

difference_random=np.zeros((1000,1))
difference_random=data_random_result-real_random
idx=np.where(difference_random==0)  
idx0=idx[0]
Same_tran=len(idx0)/len(real_random)
print('Accuracy Neural network of random data:', Same_tran)





###chose natural_results by ANN== 1,2,3


idxd1=np.where( (data_random_result==1)) 
idxd10=idxd1[0]

idxd2=np.where((data_random_result==2)) 
idxd20=idxd2[0]

#idxd3=(np.where(difference_total!=0) and np.where(Naturedata_result==3)) 

idxd3=np.where((data_random_result==3)) 
idxd30=idxd3[0]


#########

#-------------------------------------------------------------------------------

###show the results by figures
MgO=np.zeros((1000,1))
for i in range(0,1000):   
   MgO[i] = datar.iloc[i+1,7] 
   



fcmsd=np.zeros((1000,1))
for i in range(0,1000):   
   fcmsd[i] = datar.iloc[i+1,43] 


###############################
plt.figure()


ax2=plt.subplot(1,1,1)

xx=[4,32]
yy1=[0.37,0.37]
yy2=[0.05,0.05]
plt.plot(xx,yy1,'b--',alpha=0.3,linewidth=0.5)
plt.plot(xx,yy2,'b--',alpha=0.3,linewidth=0.5)

l1=plt.scatter(MgO[idxd30],fcmsd[idxd30],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
#plt.scatter(Mg[idxd0],fcmsd[idxd0],marker='+',c='r',edgecolors='b',s=15,linewidth=0.5)

l2=plt.scatter(MgO[idxd20],fcmsd[idxd20],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)

plt.scatter(MgO[idxd10],fcmsd[idxd10],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=15,linewidth=0.5)



rect = plt.Rectangle((8,-0.3),12,0.8,fill='False',facecolor='none',edgecolor='k')
ax2.add_patch(rect)

#plt.legend([l3,l4],['ANN_T','ANN_M'], loc='upper center',fontsize=7)

plt.xlabel('MgO',fontsize=12)          
plt.ylabel('FCKANTMS',fontsize=12) 
plt.xlim(4, 32)
plt.ylim(-2, 3)

plt.figtext(0.9,0.9,'(b)',color='black',fontsize=12)

plt.savefig('random_data_compare',dpi=300)

#######input emeishan
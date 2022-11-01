#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 14:43:56 2021

@author: lilucheng
"""
#here show two cases studies with ANN


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



#D=X
#idq=np.where((D[:,0]<30) | (D[:,0]>65))  
#idq0=idq[0]
#newX=np.delete(D,idq0,0)

#newy=np.delete(y,idq0)
#newy=newy.reshape(-1,1)

newX=X
newy=y

X_train, X_test, y_train, y_test = train_test_split(newX, newy, train_size=0.8, random_state = 0)


clf = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 20), random_state=1)


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
difference_total=np.zeros((915,1))
difference_total=newy-y_result
idx=np.where(difference_total==0)  
idx0=idx[0]
Same_alldata=len(idx0)/len(y_result)
print('Accuracy Neural network of all data:', Same_alldata)



#------------------------------------------------------------------------------------------
#read natural case data   Hawoii volcano

data2 = pd.read_excel('natural_case_noref.xlsx',header = None,skipfooter= 1,index_col=1)

#train data determined from dataframe
Naturedata = np.zeros((763,10))  
for i in range(0,763):  
 for j in range(0,10):  
   Naturedata[i][j] = data2.iloc[i+1,j+1]  
   
Naturedata_result=clf.predict(Naturedata)

Naturedata_result=Naturedata_result.reshape(-1,1)

#------



###chose natural_results by ANN== 1,2,3


idxd1=np.where( (Naturedata_result==1)) 
idxd10=idxd1[0]

idxd2=np.where((Naturedata_result==2)) 
idxd20=idxd2[0]

#idxd3=(np.where(difference_total!=0) and np.where(Naturedata_result==3)) 

idxd3=np.where((Naturedata_result==3)) 
idxd30=idxd3[0]


#########



###show the results by figures
MgO=np.zeros((763,1))
for i in range(0,763):   
   MgO[i] = data2.iloc[i+1,7] 
   
#here mg#
Mg=np.zeros((763,1))
for i in range(0,763):   
   Mg[i] = data2.iloc[i+1,16] 
   
   
cati=np.zeros((763,1))
for i in range(0,763):   
   cati[i] = data2.iloc[i+1,17] 
   
   
sica=np.zeros((763,1)) 
for i in range(0,763):   
   sica[i] = data2.iloc[i+1,18] 



fcmsd=np.zeros((763,1))
for i in range(0,763):   
   fcmsd[i] = data2.iloc[i+1,19] 


###############################
plt.figure()


plt.subplots_adjust(left=0.17, right=0.95,bottom=0.12,top=0.95, wspace=0, hspace=0)
ax1=plt.subplot(2,2,1)

xx=[0,1]
yy1=[0.37,0.37]
yy2=[0.05,0.05]
plt.plot(xx,yy1,'b--',alpha=0.3,linewidth=0.5)
plt.plot(xx,yy2,'b--',alpha=0.3,linewidth=0.5)




l1=plt.scatter(Mg[idxd30],fcmsd[idxd30],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
#plt.scatter(Mg[idxd0],fcmsd[idxd0],marker='+',c='r',edgecolors='b',s=15,linewidth=0.5)

l2=plt.scatter(Mg[idxd20],fcmsd[idxd20],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)

l3=plt.scatter(Mg[idxd10],fcmsd[idxd10],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=15,linewidth=0.5)



rect = plt.Rectangle((0.65,-1),0.2,1.55,fill='False',facecolor='none',edgecolor='k')

ax1.add_patch(rect)


#plt.legend([l1,l2],['Both_M','ANN_P'], loc='upper center',fontsize=7)

plt.figtext(0.18,0.9,'Mafic',color='black',fontsize=8)
plt.figtext(0.18,0.746,'Mafic \nand Transitional',color='black',fontsize=8)



plt.figtext(0.18,0.63,'Mafic, Transitional',color='black',fontsize=8)
plt.figtext(0.18,0.58,'and Peridotitic',color='black',fontsize=8)







plt.ylabel('FCKANTMS',fontsize=10) 
plt.xlim(0.4, 0.95)
plt.ylim(-1.05, 1.15)
plt.xticks([])


plt.figtext(0.51,0.9,'(a)',color='black',fontsize=12)
#-------------------------------


#-------------------------------
ax4=plt.subplot(2,2,2)

xx=[0,4]
yy1=[0.37,0.37]
yy2=[0.05,0.05]
plt.plot(xx,yy1,'b--',alpha=0.3,linewidth=0.5)
plt.plot(xx,yy2,'b--',alpha=0.3,linewidth=0.5)

l1=plt.scatter(cati[idxd30],fcmsd[idxd30],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
#plt.scatter(Mg[idxd0],fcmsd[idxd0],marker='+',c='r',edgecolors='b',s=15,linewidth=0.5)

l2=plt.scatter(cati[idxd20],fcmsd[idxd20],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)

l3=plt.scatter(cati[idxd10],fcmsd[idxd10],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=15,linewidth=0.5)







rect = plt.Rectangle((1,-1),2.5,1.55,fill='False',facecolor='none',edgecolor='k')

ax4.add_patch(rect)


plt.legend([l1,l2,l3],['Mafic','Transitional','Peridotite'], loc='lower center',fontsize=7)


#plt.xlabel('ln(CaO/TiO2)',fontsize=12)   
#plt.xlabel('ln(CaO/Ti$\mathregular{O_2}$)',fontsize=10)          
       
#plt.ylabel('FCKANTMS',fontsize=10) 

plt.xticks([])
plt.yticks([])

plt.xlim(0.5, 4)
plt.ylim(-1.05, 1.15)

plt.figtext(0.9,0.9,'(d)',color='black',fontsize=12)

#plt.legend();
#plt.ylim(0, 100)
#plt.figtext(0.02,0.975,'(a)',color='black',fontsize=15)
#plt.figtext(0.82,0.96,'1-1-2',color='black',fontsize=13)

#plt.xticks(fontsize=12) 
#plt.yticks(fontsize=12) 

#plt.xlabel('An(%)',fontsize=14)          
#plt.ylabel('Frequency(%)',fontsize=14) 
#plt.xticks(np.arange(0.0,110.0,10))

#plt.figtext(0.5,0.83,'S_ID*RA=74%',color='black',fontsize=12)

#######input emeishan
#-----------------------------------------------------------------------------
data_emeishan = pd.read_excel('emeishan.xlsx',header = None,skipfooter= 1,index_col=1)

#change 1075 into 1240

#train data determined from dataframe
Emeishandata = np.zeros((1240,10))  
for i in range(0,1240):  
 for j in range(0,10):  
   Emeishandata[i][j] = data_emeishan.iloc[i+1,j+3]  
   
Emeishandata_result=clf.predict(Emeishandata)

Emeishandata_result=Emeishandata_result.reshape(-1,1)





idxd_v1=np.where(difference_total!=0)  
idxd_v10=idxd_v1[0]

###different

idx_v1_1=np.where(Emeishandata_result==1)
idx_v1_10=idx_v1_1[0]


idx_v1_2=np.where(Emeishandata_result==2)
idx_v1_20=idx_v1_2[0]

idx_v1_3=np.where(Emeishandata_result==3)
idx_v1_30=idx_v1_3[0]

 

###show the results by figures
MgOe=np.zeros((1240,1))
for i in range(0,1240):   
   MgOe[i] = data_emeishan.iloc[i+1,9] 
   

Na2oe=np.zeros((1240,1))
for i in range(0,1240):   
   Na2oe[i] = data_emeishan.iloc[i+1,11] 
   
   
K2oe=np.zeros((1240,1))
for i in range(0,1240):   
   K2oe[i] = data_emeishan.iloc[i+1,12] 
   
   
#version1
Fc3ms=np.zeros((1240,1))
for i in range(0,1240):   
   Fc3ms[i] = data_emeishan.iloc[i+1,15] 


#version2
Fcms_v2=np.zeros((1240,1))
for i in range(0,1240):   
   Fcms_v2[i] = data_emeishan.iloc[i+1,17] 
   
   
catie=np.zeros((1240,1))
for i in range(0,1240):   
   catie[i] = data_emeishan.iloc[i+1,13] 

sicae=np.zeros((1240,1))
for i in range(0,1240):   
   sicae[i] = data_emeishan.iloc[i+1,14]    
   
   

nak=Na2oe+K2oe

#compute Mg#

mgjh=np.zeros((1240,1))
for i in range(0,1240):   
   mgjh[i] = data_emeishan.iloc[i+1,19]    
   


#-----------------------------------------------------------------------------------------

###############################




ax3=plt.subplot(2,2,3)

#plt.scatter(nak[idx_v20],Fcms_v2[idx_v20],marker='o',c='',edgecolors='k',s=15,linewidth=0.5)
#plt.scatter(nak[idx_v2_10],Fcms_v2[idx_v2_10],marker='+',c='r',edgecolors='r',s=15,linewidth=0.5)
#plt.scatter(nak[idx_v2_20],Fcms_v2[idx_v2_20],marker='x',c='g',edgecolors='g',s=15,linewidth=0.5)
#plt.scatter(nak[idx_v2_30],Fcms_v2[idx_v2_30],marker='*',c='y',edgecolors='y',s=15,linewidth=0.5)

xx=[0,2]
yy1=[0.37,0.37]
yy2=[0.05,0.05]
plt.plot(xx,yy1,'b--',alpha=0.3,linewidth=0.5)
plt.plot(xx,yy2,'b--',alpha=0.3,linewidth=0.5)




plt.scatter(mgjh[idx_v1_30],Fcms_v2[idx_v1_30],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(mgjh[idx_v1_20],Fcms_v2[idx_v1_20],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
plt.scatter(mgjh[idx_v1_10],Fcms_v2[idx_v1_10],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=15,linewidth=0.5)

     
#plt.legend([l1,l2],['Both_M','ANN_P'], loc='left center',fontsize=7)




rect = plt.Rectangle((0.65,-1),0.2,1.55,fill='False',facecolor='none',edgecolor='k')
ax3.add_patch(rect)




#plt.xlabel('ln(Si$\mathregular{O_2}$/CaO+$\mathregular{Na_2}$O+Ti$\mathregular{O_2}$)',fontsize=12)   
plt.xlabel('Mg#',fontsize=10)          
plt.ylabel('FCKANTMS',fontsize=10) 
plt.xlim(0.4, 0.95)
plt.ylim(-1.05, 1.15)






plt.figtext(0.51,0.48,'(c)',color='black',fontsize=12)


#-------------------------------------------------------------------
ax4=plt.subplot(2,2,4)

#plt.scatter(MgOe[idx_v20],Fcms_v2[idx_v20],marker='o',c='',edgecolors='k',s=15,linewidth=0.5)
#plt.scatter(MgOe[idx_v2_10],Fcms_v2[idx_v2_10],marker='+',c='r',edgecolors='r',s=15,linewidth=0.5)
#plt.scatter(MgOe[idx_v2_20],Fcms_v2[idx_v2_20],marker='x',c='g',edgecolors='g',s=15,linewidth=0.5)
#plt.scatter(MgOe[idx_v2_30],Fcms_v2[idx_v2_30],marker='*',c='y',edgecolors='y',s=15,linewidth=0.5)

xx=[0,4]
yy1=[0.37,0.37]
yy2=[0.05,0.05]
plt.plot(xx,yy1,'b--',alpha=0.3,linewidth=0.5)
plt.plot(xx,yy2,'b--',alpha=0.3,linewidth=0.5)



l1=plt.scatter(catie[idx_v1_30],Fcms_v2[idx_v1_30],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
l2=plt.scatter(catie[idx_v1_20],Fcms_v2[idx_v1_20],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
l3=plt.scatter(catie[idx_v1_10],Fcms_v2[idx_v1_10],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=15,linewidth=0.5)


#plt.legend([l3,l4],['ANN_T','ANN_M'], loc='left center',fontsize=7)



rect = plt.Rectangle((1,-1),2.5,1.55,fill='False',facecolor='none',edgecolor='k')

ax4.add_patch(rect)



#plt.xlabel('Na2O+k2O(%)',fontsize=12)     
plt.xlabel('ln(CaO/Ti$\mathregular{O_2}$)',fontsize=10)          
     
#plt.ylabel('FCKANTMS',fontsize=10) 
plt.xlim(0.5, 4)
plt.ylim(-1.05, 1.15)
plt.yticks([])

plt.figtext(0.9,0.48,'(d)',color='black',fontsize=12)

plt.savefig('fig6_two_nature_cases',dpi=300)




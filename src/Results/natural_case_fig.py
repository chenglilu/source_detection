#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:26:50 2021

@author: lilucheng
"""

#for natural case
#only show our results
#2021-3-8




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



#------------------------------------------------------------------------------------------
#read natural case data

data2 = pd.read_excel('natural_case_noref.xlsx',header = None,skipfooter= 1,index_col=1)

#train data determined from dataframe
Naturedata = np.zeros((763,10))  
for i in range(0,763):  
 for j in range(0,10):  
   Naturedata[i][j] = data2.iloc[i+1,j+1]  
   
Naturedata_result=clf.predict(Naturedata)

Naturedata_result=Naturedata_result.reshape(-1,1)

#------
#result compare
Previousresult=np.zeros((763,1)) 
for i in range(0,763):   
   Previousresult[i] = data2.iloc[i+1,14]  

difference_total=np.zeros((763,1))
difference_total=Previousresult-Naturedata_result


idx=np.where(difference_total==0)  
idx0=idx[0]
Same_rate=len(idx0)/len(Previousresult)
print('Same number of all the data',Same_rate)   



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


plt.subplots_adjust(left=0.17, right=0.95,bottom=0.12,top=0.95, wspace=0.4, hspace=0.35)
ax1=plt.subplot(2,2,1)

xx=[0.38,0.92]
yy1=[0.37,0.37]
yy2=[0.05,0.05]
plt.plot(xx,yy1,'b--',alpha=0.3,linewidth=0.5)
plt.plot(xx,yy2,'b--',alpha=0.3,linewidth=0.5)




l1=plt.scatter(Mg[idxd30],fcmsd[idxd30],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
#plt.scatter(Mg[idxd0],fcmsd[idxd0],marker='+',c='r',edgecolors='b',s=15,linewidth=0.5)

l2=plt.scatter(Mg[idxd20],fcmsd[idxd20],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)

plt.scatter(Mg[idxd10],fcmsd[idxd10],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=15,linewidth=0.5)



rect = plt.Rectangle((0.65,-0.3),0.15,0.8,fill='False',facecolor='none',edgecolor='k')
ax1.add_patch(rect)


#plt.legend([l1,l2],['Both_M','ANN_P'], loc='upper center',fontsize=7)

plt.figtext(0.18,0.77,'Mafic',color='black',fontsize=8)
plt.figtext(0.18,0.63,'Mafic and Transitional',color='black',fontsize=8)



plt.xlabel('Mg#',fontsize=12)          
plt.ylabel('FCKANTMS',fontsize=12) 
plt.xlim(0.38, 0.92)

plt.figtext(0.45,0.9,'(a)',color='black',fontsize=12)
#-------------------------------
ax2=plt.subplot(2,2,2)

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
plt.figtext(0.9,0.9,'(b)',color='black',fontsize=12)

#-------------------------------
ax3=plt.subplot(2,2,3)

xx=[0.8,2]
yy1=[0.37,0.37]
yy2=[0.05,0.05]
plt.plot(xx,yy1,'b--',alpha=0.3,linewidth=0.5)
plt.plot(xx,yy2,'b--',alpha=0.3,linewidth=0.5)

l1=plt.scatter(sica[idxd30],fcmsd[idxd30],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
#plt.scatter(Mg[idxd0],fcmsd[idxd0],marker='+',c='r',edgecolors='b',s=15,linewidth=0.5)

l2=plt.scatter(sica[idxd20],fcmsd[idxd20],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)

plt.scatter(sica[idxd10],fcmsd[idxd10],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=15,linewidth=0.5)






rect = plt.Rectangle((1.1,-0.3),0.49,0.8,fill='False',facecolor='none',edgecolor='k')
ax3.add_patch(rect)


#plt.xlabel('ln(SiO2/CaO+Na2O+TiO2)',fontsize=12)        
plt.xlabel('ln(Si$\mathregular{O_2}$/CaO+$\mathregular{Na_2}$O+Ti$\mathregular{O_2}$)',fontsize=12)   
  
plt.ylabel('FCKANTMS',fontsize=12) 
plt.xlim(0.8, 2)

plt.figtext(0.45,0.43,'(c)',color='black',fontsize=12)

#-------------------------------
ax4=plt.subplot(2,2,4)

xx=[0.8,2]
yy1=[0.37,0.37]
yy2=[0.05,0.05]
plt.plot(xx,yy1,'b--',alpha=0.3,linewidth=0.5)
plt.plot(xx,yy2,'b--',alpha=0.3,linewidth=0.5)

l1=plt.scatter(cati[idxd30],fcmsd[idxd30],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
#plt.scatter(Mg[idxd0],fcmsd[idxd0],marker='+',c='r',edgecolors='b',s=15,linewidth=0.5)

l2=plt.scatter(cati[idxd20],fcmsd[idxd20],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)

l3=plt.scatter(cati[idxd10],fcmsd[idxd10],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=15,linewidth=0.5)






plt.legend([l1,l2,l3],['Mafic','Transitional','Peridotite'], loc='lower left',fontsize=7, ncol=1)

rect = plt.Rectangle((1.5,-0.3),1.5,0.8,fill='False',facecolor='none',edgecolor='k')
ax4.add_patch(rect)



#plt.xlabel('ln(CaO/TiO2)',fontsize=12)   
plt.xlabel('ln(CaO/Ti$\mathregular{O_2}$)',fontsize=12)          
       
plt.ylabel('FCKANTMS',fontsize=12) 
plt.xlim(0.8, 2)
plt.figtext(0.9,0.43,'(d)',color='black',fontsize=12)

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
plt.savefig('Mauna_result_without_compare',dpi=300)

#######input emeishan
#-----------------------------------------------------------------------------
data_emeishan = pd.read_excel('emeishan.xlsx',header = None,skipfooter= 1,index_col=1)

#train data determined from dataframe
Emeishandata = np.zeros((1075,10))  
for i in range(0,1075):  
 for j in range(0,10):  
   Emeishandata[i][j] = data_emeishan.iloc[i+1,j+3]  
   
Emeishandata_result=clf.predict(Emeishandata)

Emeishandata_result=Emeishandata_result.reshape(-1,1)



#------
#result compare
#result compare
Previousemeishan=np.zeros((1075,1)) 
for i in range(0,1075):   
   Previousemeishan[i] = data_emeishan.iloc[i+1,16]  

difference_total=np.zeros((1075,1))
difference_total=Previousemeishan-Emeishandata_result


idx_v1=np.where(difference_total==0)  
idx_v10=idx_v1[0]
Same_rate_v1=len(idx_v10)/len(Previousemeishan)
print('Same number of emeishan_version1',Same_rate_v1)  


idxd_v1=np.where(difference_total!=0)  
idxd_v10=idxd_v1[0]

###different

idx_v1_1=np.where(Emeishandata_result==1)
idx_v1_10=idx_v1_1[0]


idx_v1_2=np.where(Emeishandata_result==2)
idx_v1_20=idx_v1_2[0]

idx_v1_3=np.where(Emeishandata_result==3)
idx_v1_30=idx_v1_3[0]

###histogram
plt.figure()


plt.hist(Emeishandata_result, bins=[1,2,3,4],density=True,     
                weights=None, cumulative=False, bottom=None,     
                histtype='bar', align='left', orientation='vertical' )
plt.xlabel('Group',fontsize=14)
plt.ylabel('Presentage',fontsize=14)


 

###show the results by figures
MgOe=np.zeros((1075,1))
for i in range(0,1075):   
   MgOe[i] = data_emeishan.iloc[i+1,9] 
   

Na2oe=np.zeros((1075,1))
for i in range(0,1075):   
   Na2oe[i] = data_emeishan.iloc[i+1,11] 
   
   
K2oe=np.zeros((1075,1))
for i in range(0,1075):   
   K2oe[i] = data_emeishan.iloc[i+1,12] 
   
   
#version1
Fc3ms=np.zeros((1075,1))
for i in range(0,1075):   
   Fc3ms[i] = data_emeishan.iloc[i+1,15] 


#version2
Fcms_v2=np.zeros((1075,1))
for i in range(0,1075):   
   Fcms_v2[i] = data_emeishan.iloc[i+1,17] 
   
   
catie=np.zeros((1075,1))
for i in range(0,1075):   
   catie[i] = data_emeishan.iloc[i+1,13] 

sicae=np.zeros((1075,1))
for i in range(0,1075):   
   sicae[i] = data_emeishan.iloc[i+1,14]    
   
   

nak=Na2oe+K2oe



#----------------------------------------------------------------------------------------
######compare vs FCKANTMS   version2 so call Previousemeishan2

Previousemeishan2=np.zeros((1075,1)) 
for i in range(0,1075):   
   Previousemeishan2[i] = data_emeishan.iloc[i+1,18]  

difference_total2=np.zeros((1075,1))
difference_total2=Previousemeishan2-Emeishandata_result



idx_v2=np.where(difference_total2==0)  
idx_v20=idx_v2[0]
Same_rate_v2=len(idx_v20)/len(Previousemeishan2)
print('Same number of emeishan version2',Same_rate_v2)  



###different
idxd_v2=np.where(difference_total2!=0) 
idxd_v20=idxd_v2[0]



idx_v2_1=np.where((difference_total2!=0) & (Emeishandata_result==1)) 
idx_v2_10=idx_v2_1[0]


idx_v2_2=np.where((difference_total2!=0) & (Emeishandata_result==2)) 
idx_v2_20=idx_v2_2[0]

idx_v2_3=np.where((difference_total2!=0) & (Emeishandata_result==3)) 
idx_v2_30=idx_v2_3[0]







#-----------------------------------------------------------------------------------------

###############################
plt.figure()


plt.subplots_adjust(left=0.17, right=0.95,bottom=0.12,top=0.95, wspace=0.4, hspace=0.35)
plt.subplot(2,2,1)






plt.scatter(nak[idx_v1_30],Fc3ms[idx_v1_30],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(nak[idx_v1_20],Fc3ms[idx_v1_20],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
plt.scatter(nak[idx_v1_10],Fc3ms[idx_v1_10],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=15,linewidth=0.5)

xx=[0,5]
yy2=[0.65,0.65]
plt.plot(xx,yy2,'k--',alpha=0.3,linewidth=0.5)


plt.xlabel('$\mathregular{Na_2}$O+$\mathregular{K_2}$O(%)',fontsize=12)          
plt.ylabel('FC3MS',fontsize=12) 
plt.xlim(0, 5)

plt.figtext(0.17,0.9,'(a)',color='black',fontsize=12)


plt.subplot(2,2,2)



plt.scatter(MgOe[idx_v1_30],Fc3ms[idx_v1_30],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(MgOe[idx_v1_20],Fc3ms[idx_v1_20],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
plt.scatter(MgOe[idx_v1_10],Fc3ms[idx_v1_10],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=15,linewidth=0.5)


xx=[4,24]
yy2=[0.65,0.65]
plt.plot(xx,yy2,'k--',alpha=0.3,linewidth=0.5)

plt.xlabel('MgO(%)',fontsize=12)          
plt.ylabel('FC3MS',fontsize=12) 
plt.xlim(4, 24)

plt.annotate('Below: Peridotite', xy=(12, 0.65), xycoords='data',
                xytext=(10, -30), textcoords='offset points',
                  size=8,
                arrowprops=dict(arrowstyle="->")
                )   
                


plt.figtext(0.63,0.9,'(b)',color='black',fontsize=12)


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




plt.scatter(sicae[idx_v1_30],Fcms_v2[idx_v1_30],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(sicae[idx_v1_20],Fcms_v2[idx_v1_20],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
plt.scatter(sicae[idx_v1_10],Fcms_v2[idx_v1_10],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=15,linewidth=0.5)

     
#plt.legend([l1,l2],['Both_M','ANN_P'], loc='left center',fontsize=7)




rect = plt.Rectangle((1.1,-0.3),0.49,0.8,fill='False',facecolor='none',edgecolor='k')
ax3.add_patch(rect)




plt.xlabel('ln(Si$\mathregular{O_2}$/CaO+$\mathregular{Na_2}$O+Ti$\mathregular{O_2}$)',fontsize=12)   
#plt.xlabel('MgO',fontsize=12)          
plt.ylabel('FCKANTMS',fontsize=12) 
plt.xlim(0, 2)


plt.figtext(0.22,0.41,'Mafic',color='black',fontsize=7)
plt.figtext(0.18,0.355,'Mafic',color='black',fontsize=7)
plt.figtext(0.18,0.315,'and Transitional',color='black',fontsize=7)

plt.figtext(0.18,0.2,'Mafic, Transitional',color='black',fontsize=7)
plt.figtext(0.18,0.16,'and Peridotite',color='black',fontsize=7)





plt.figtext(0.17,0.43,'(c)',color='black',fontsize=12)

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



rect = plt.Rectangle((1.5,-0.3),1.5,0.8,fill='False',facecolor='none',edgecolor='k')
ax4.add_patch(rect)


plt.legend([l1,l2,l3],['Mafic','Transitional','Peridotite'], loc='lower left',fontsize=7, ncol=1)

#plt.xlabel('Na2O+k2O(%)',fontsize=12)     
plt.xlabel('ln(CaO/Ti$\mathregular{O_2}$)',fontsize=12)          
     
plt.ylabel('FCKANTMS',fontsize=12) 
plt.xlim(0, 4)

plt.figtext(0.63,0.43,'(d)',color='black',fontsize=12)

plt.savefig('emeishan_result_without_campare',dpi=300)


#-----------------------------------------------------------------------------------------
#emeishan picrite

data_emeishan_p = pd.read_excel('emeishan_picrite.xlsx',header = None,skipfooter= 1,index_col=1)

#train data determined from dataframe
Emeishandatap = np.zeros((67,10))  
for i in range(0,67):  
 for j in range(0,10):  
   Emeishandatap[i][j] = data_emeishan_p.iloc[i+1,j+1]  
   
Emeishandatap_result=clf.predict(Emeishandatap)

Emeishandatap_result=Emeishandatap_result.reshape(-1,1)



###labels

idx_v1_1p=np.where(Emeishandatap_result==1)
idx_v1_10p=idx_v1_1p[0]


idx_v1_2p=np.where(Emeishandatap_result==2)
idx_v1_20p=idx_v1_2p[0]

idx_v1_3p=np.where(Emeishandatap_result==3)
idx_v1_30p=idx_v1_3p[0]



###show the results by figures
MgOp=np.zeros((67,1))
for i in range(0,67):   
   MgOp[i] = data_emeishan_p.iloc[i+1,7] 
   


#version2
Fcms_v2p=np.zeros((67,1))
for i in range(0,67):   
   Fcms_v2p[i] = data_emeishan_p.iloc[i+1,14] 
   
   
catip=np.zeros((67,1))
for i in range(0,67):   
   catip[i] = data_emeishan_p.iloc[i+1,12] 

sicap=np.zeros((67,1))
for i in range(0,67):   
   sicap[i] = data_emeishan_p.iloc[i+1,13]    
   
   




#----------------------------------------------------------------------------------------
######compare vs FCKANTMS 

plt.figure()





#plt.scatter(MgOe[idx_v20],Fcms_v2[idx_v20],marker='o',c='',edgecolors='k',s=15,linewidth=0.5)
#plt.scatter(MgOe[idx_v2_10],Fcms_v2[idx_v2_10],marker='+',c='r',edgecolors='r',s=15,linewidth=0.5)
#plt.scatter(MgOe[idx_v2_20],Fcms_v2[idx_v2_20],marker='x',c='g',edgecolors='g',s=15,linewidth=0.5)
#plt.scatter(MgOe[idx_v2_30],Fcms_v2[idx_v2_30],marker='*',c='y',edgecolors='y',s=15,linewidth=0.5)
ax4=plt.subplot(1,1,1)
xx=[0,4]
yy1=[0.37,0.37]
yy2=[0.05,0.05]
plt.plot(xx,yy1,'b--',alpha=0.3,linewidth=0.5)
plt.plot(xx,yy2,'b--',alpha=0.3,linewidth=0.5)



l1=plt.scatter(catip[idx_v1_30p],Fcms_v2p[idx_v1_30p],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
l2=plt.scatter(catip[idx_v1_20p],Fcms_v2p[idx_v1_20p],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
l3=plt.scatter(catip[idx_v1_10p],Fcms_v2p[idx_v1_10p],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=15,linewidth=0.5)


#plt.legend([l3,l4],['ANN_T','ANN_M'], loc='left center',fontsize=7)



rect = plt.Rectangle((1.5,-0.3),1.5,0.8,fill='False',facecolor='none',edgecolor='k')
ax4.add_patch(rect)


plt.legend([l1,l2,l3],['Mafic','Transitional','Peridotite'], loc='lower left',fontsize=7, ncol=1)

#plt.xlabel('Na2O+k2O(%)',fontsize=12)     
plt.xlabel('ln(CaO/Ti$\mathregular{O_2}$)',fontsize=12)          
     
plt.ylabel('FCKANTMS',fontsize=12) 
plt.xlim(0, 4)

plt.figtext(0.63,0.43,'(d)',color='black',fontsize=12)





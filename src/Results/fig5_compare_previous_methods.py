#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:44:22 2021

@author: lilucheng
"""
#produce a new figure 5 
#compare the difference method on the database
#(a) CaO-MgO
#(b) FC3MS
#(c),(d)  FCMS

#reset -f  #clear all variable


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

X_train, X_test, y_train, y_test = train_test_split(newX, newy, train_size=0.9, random_state = 0)


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
# mgo,mg#,cati,sica

   
mg=mgjh
  
    
fcms=np.zeros((915,1))
for i in range(0,915):   
   fcms[i] = data.iloc[i+1,21]      

   
cati=np.zeros((915,1))
for i in range(0,915):   
   cati[i] = data.iloc[i+1,22]  

sica=np.zeros((915,1))
for i in range(0,915):   
   sica[i] = data.iloc[i+1,23]  

   
   
   
   #record
N_inbox_total=np.zeros(4)
N_outbox_total=np.zeros(4)

Accuracy_inbox_total=np.zeros(4)
Accuracy_outbox_total=np.zeros(4)
   
#-------------------------------------------------------
#start to plot


plt.figure()



plt.subplots_adjust(left=0.1, right=0.95,bottom=0.13,top=0.98, wspace=0.3, hspace=0.5)
ax1=plt.subplot(2,2,1)



l1=plt.scatter(mgo[idx3r0],cao[idx3r0],marker='^',c='',edgecolors='0.5',s=30,linewidth=0.5)
l2=plt.scatter(mgo[idx2r0],cao[idx2r0],marker='s',c='limegreen',edgecolors='k',s=30,linewidth=0.5)
l3=plt.scatter(mgo[idx1r0],cao[idx1r0],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=30,linewidth=0.5)



#plt.scatter(mgo[idx3w0],cao[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=30,linewidth=0.8)
#plt.scatter(mgo[idx2w0],cao[idx2w0],marker='s',edgecolors='orangered',facecolor='gold',s=30,linewidth=0.8)
#plt.scatter(mgo[idx1w0],cao[idx1w0],marker='x',c='orangered',s=30,linewidth=0.8)


plt.scatter(mgo[idx3w0],cao[idx3w0],marker='^',edgecolors='k',facecolor='orangered',s=30,linewidth=0.5)
plt.scatter(mgo[idx2w0],cao[idx2w0],marker='s',edgecolors='k',facecolor='orangered',s=30,linewidth=0.5)
plt.scatter(mgo[idx1w0],cao[idx1w0],marker='x',c='orangered',s=30,linewidth=0.8)




plt.legend([l1,l2,l3],['Mafic R', 'Transitional R', 'Peridotitic R'], loc='upper right',fontsize=5)

#cao=13.81-0.274MgO  from Herzberg & Asimow 2008

#rect = plt.Rectangle((0.65,42),0.2,15,fill='False',facecolor='none',edgecolor='k')
#plt1.add_patch(rect)



rect = plt.Rectangle((6,4),14,12,fill='False',facecolor='none',edgecolor='k')
ax1.add_patch(rect)



plt.xlabel('MgO(wt%)',fontsize=12)          
plt.ylabel('CaO(wt%)',fontsize=12) 

#plt.figtext(0.28,0.9,'(a)',color='black',fontsize=12)


#plt.legend(['Mafic R', 'Transitional R', 'Peridotite R','Mafic W','Transitional W','Peridotite W'], loc='top right',fontsize=8)
x1=[0,45]
y1=[13.81,1.48]

plt.xlim(0,45)
plt.text(27,15,'Peridotite',fontsize=8)
plt.text(20,0.7,'Pyroxenite ',fontsize=8)


plt.plot(x1,y1,'b--',alpha=0.5,linewidth=1)

plt.text(0.4,24.5,'(a)',fontsize=12)


#----------------------------------------------------------------------
#determine the accuracy in the box
g1=np.where((cao>=4) & (cao<=16) & (mgo>=6) & (mgo<=20))    #right
g10=g1[0]
diff_in_box=difference_total[g10]

N_diff=np.where(diff_in_box==0)  
N_diff0=N_diff[0]
Same_in_box=len(N_diff0)/len(diff_in_box)
print('Number in box fig1:    ',len(diff_in_box))   
print('Accuracy in box fig1:    ',Same_in_box)   

idx=np.where(difference_total==0)  
idx0=idx[0]
N_total_same=len(idx0)
Same_out_box=(N_total_same-len(N_diff0))/(915-len(diff_in_box))

print('Number out box fig1:    ',915-len(diff_in_box))   
print('Accuracy out box fig1:    ',Same_out_box)   

print('----------------------------------------------------------')   

N_inbox_total[0]=len(diff_in_box)
N_outbox_total[0]=915-len(diff_in_box)

Accuracy_inbox_total[0]=Same_in_box
Accuracy_outbox_total[0]=Same_out_box

value1=100*Same_out_box

plt.figtext(0.35,0.75,"%.0f%%"  % value1)


#----------------------------------------------------------------------




#-------------------------------compare to FC3MS

ax2=plt.subplot(2,2,2)


l1=plt.scatter(mgo[idx3r0],fc3ms[idx3r0],marker='^',c='',edgecolors='0.5',s=30,linewidth=0.5)
l2=plt.scatter(mgo[idx2r0],fc3ms[idx2r0],marker='s',c='limegreen',edgecolors='k',s=30,linewidth=0.5)
l3=plt.scatter(mgo[idx1r0],fc3ms[idx1r0],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=30,linewidth=0.5)

plt.scatter(mgo[idx3r0],fc3ms[idx3r0],marker='^',c='',edgecolors='0.5',s=30,linewidth=0.5)


l4=plt.scatter(mgo[idx3w0],fc3ms[idx3w0],marker='^',edgecolors='k',facecolor='orangered',s=30,linewidth=0.5)
l5=plt.scatter(mgo[idx2w0],fc3ms[idx2w0],marker='s',edgecolors='k',facecolor='orangered',s=30,linewidth=0.5)
l6=plt.scatter(mgo[idx1w0],fc3ms[idx1w0],marker='x',c='orangered',s=30,linewidth=0.5)



#cao=13.81-0.274MgO  from Herzberg & Asimow 2008

#rect = plt.Rectangle((0.65,42),0.2,15,fill='False',facecolor='none',edgecolor='k')
#plt1.add_patch(rect)

rect = plt.Rectangle((6,-0.8),14,1.37,fill='False',facecolor='none',edgecolor='k')
ax2.add_patch(rect)




#----------------------------------------------------------------------
#determine the accuracy in the box
g2=np.where((fc3ms>=-0.8) & (fc3ms<=0.57) & (mgo>=6) & (mgo<=20))    #right
g20=g2[0]
diff_in_box=difference_total[g20]

N_diff=np.where(diff_in_box==0)  
N_diff0=N_diff[0]
Same_in_box=len(N_diff0)/len(diff_in_box)
print('Number in box fig2:    ',len(diff_in_box))   
print('Accuracy in box fig2:    ',Same_in_box)   

idx=np.where(difference_total==0)  
idx0=idx[0]
N_total_same=len(idx0)
Same_out_box=(N_total_same-len(N_diff0))/(915-len(diff_in_box))

print('Number out box fig2:    ',915-len(diff_in_box))   
print('Accuracy out box fig2:    ',Same_out_box)   

print('----------------------------------------------------------')   

N_inbox_total[1]=len(diff_in_box)
N_outbox_total[1]=915-len(diff_in_box)

Accuracy_inbox_total[1]=Same_in_box
Accuracy_outbox_total[1]=Same_out_box

value2=100*Same_out_box

plt.figtext(0.82,0.7,"%.0f%%"  % value2)


#----------------------------------------------------------------------




plt.xlabel('MgO(wt%)',fontsize=12)          
plt.ylabel('FC3MS',fontsize=12) 

#plt.figtext(0.28,0.9,'(a)',color='black',fontsize=12)

plt.legend([l4,l5,l6],['Mafic W','Transitional W','Peridotitic W'], loc='upper right',fontsize=6)
x1=[0,45]

yy2=[0.65,0.65]
plt.plot(x1,yy2,'b--',alpha=0.5,linewidth=1)

plt.text(27,0.86,'Pyroxenite melt',fontsize=8)
plt.text(27,0.12,'Peridotite melt',fontsize=8)



plt.xlim(0,45)

#plt.ylim(-3.2,4)


plt.text(0.5,3.9,'(b)',fontsize=12)










#-------------------------------compare to FCMS

ax3=plt.subplot(2,2,3)




xx=[0,1]
yy1=[0.37,0.37]
yy2=[0.05,0.05]
plt.plot(xx,yy1,'b--',alpha=0.3,linewidth=0.5)
plt.plot(xx,yy2,'b--',alpha=0.3,linewidth=0.5)



l1=plt.scatter(mg[idx3r0],fcms[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
l2=plt.scatter(mg[idx2r0],fcms[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
l3=plt.scatter(mg[idx1r0],fcms[idx1r0],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=15,linewidth=0.5)



l4=plt.scatter(mg[idx3w0],fcms[idx3w0],marker='^',edgecolors='k',facecolor='orangered',s=15,linewidth=0.5)
l5=plt.scatter(mg[idx2w0],fcms[idx2w0],marker='s',edgecolors='k',facecolor='orangered',s=15,linewidth=0.5)
l6=plt.scatter(mg[idx1w0],fcms[idx1w0],marker='x',c='orangered',s=15,linewidth=0.5)


#ax = fig.add_subplot(111)

rect = plt.Rectangle((0.65,-1),0.2,1.55,fill='False',facecolor='none',edgecolor='k')
ax3.add_patch(rect)

#----------------------------------------------------------------------
#determine the accuracy in the box
g3=np.where((mg>=0.65) & (mg<=0.85) & (fcms>=-1) & (fcms<=0.55))    #right
g30=g3[0]
diff_in_box=difference_total[g30]

N_diff=np.where(diff_in_box==0)  
N_diff0=N_diff[0]
Same_in_box=len(N_diff0)/len(diff_in_box)
print('Number in box fig3:    ',len(diff_in_box))   
print('Accuracy in box fig3:    ',Same_in_box)   

idx=np.where(difference_total==0)  
idx0=idx[0]
N_total_same=len(idx0)
Same_out_box=(N_total_same-len(N_diff0))/(915-len(diff_in_box))

print('Number out box fig3:    ',915-len(diff_in_box))   
print('Accuracy out box fig3:    ',Same_out_box)   

print('----------------------------------------------------------')   

N_inbox_total[2]=len(diff_in_box)
N_outbox_total[2]=915-len(diff_in_box)

Accuracy_inbox_total[2]=Same_in_box
Accuracy_outbox_total[2]=Same_out_box

value3=100*Same_out_box

plt.figtext(0.28,0.4,"%.0f%%"  % value3)


#----------------------------------------------------------------------







plt.figtext(0.12,0.38,'Mafic',color='black',fontsize=7)
plt.figtext(0.12,0.318,'M & T',color='black',fontsize=7)
plt.figtext(0.12,0.252,'Mafic,',color='black',fontsize=7)
plt.figtext(0.119,0.22,'Transitional',color='black',fontsize=7)
plt.figtext(0.12,0.19,'and Peridotitic',color='black',fontsize=7)




plt.xlabel('Mg#',fontsize=11)          
plt.ylabel('FCKANTMS',fontsize=11) 

plt.xlim(xx)

plt.figtext(0.105,0.42,'(c)',color='black',fontsize=12)








#-------------------------------compare to FCMS

ax4=plt.subplot(2,2,4)



xx=[-2,6]
yy1=[0.37,0.37]
yy2=[0.05,0.05]
plt.plot(xx,yy1,'b--',alpha=0.3,linewidth=0.5)
plt.plot(xx,yy2,'b--',alpha=0.3,linewidth=0.5)



plt.scatter(cati[idx3r0],fcms[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(cati[idx2r0],fcms[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
plt.scatter(cati[idx1r0],fcms[idx1r0],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=15,linewidth=0.5)



plt.scatter(cati[idx3w0],fcms[idx3w0],marker='^',edgecolors='k',facecolor='orangered',s=15,linewidth=0.5)
plt.scatter(cati[idx2w0],fcms[idx2w0],marker='s',edgecolors='k',facecolor='orangered',s=15,linewidth=0.5)
plt.scatter(cati[idx1w0],fcms[idx1w0],marker='x',c='orangered',s=15,linewidth=0.5)




rect = plt.Rectangle((1,-1),2.5,1.55,fill='False',facecolor='none',edgecolor='k')
ax4.add_patch(rect)



#----------------------------------------------------------------------
#determine the accuracy in the box
g4=np.where((cati>=1) & (cati<=3.5) & (fcms>=-1) & (fcms<=0.55))    #right
g40=g4[0]
diff_in_box=difference_total[g40]

N_diff=np.where(diff_in_box==0)  
N_diff0=N_diff[0]
Same_in_box=len(N_diff0)/len(diff_in_box)
print('Number in box fig4:    ',len(diff_in_box))   
print('Accuracy in box fig4:    ',Same_in_box)   

idx=np.where(difference_total==0)  
idx0=idx[0]
N_total_same=len(idx0)
Same_out_box=(N_total_same-len(N_diff0))/(915-len(diff_in_box))

print('Number out box fig4:    ',915-len(diff_in_box))   
print('Accuracy out box fig4:    ',Same_out_box)   

print('----------------------------------------------------------')   

N_inbox_total[3]=len(diff_in_box)
N_outbox_total[3]=915-len(diff_in_box)

Accuracy_inbox_total[3]=Same_in_box
Accuracy_outbox_total[3]=Same_out_box

value3=100*Same_out_box

plt.figtext(0.65,0.25,"%.0f%%"  % value3)


#----------------------------------------------------------------------





plt.xlabel('ln(CaO/Ti$\mathregular{O_2}$)',fontsize=12)          
plt.xlim(xx)


plt.figtext(0.59,0.42,'(d)',color='black',fontsize=12)


plt.ylabel('FCKANTMS',fontsize=11) 






#----save fig
plt.savefig('fig5_compare_previous_checkdata',dpi=300)

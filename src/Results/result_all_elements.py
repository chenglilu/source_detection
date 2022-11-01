#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 13:07:25 2021

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
   
#record
N_inbox_total=np.zeros(9)
N_outbox_total=np.zeros(9)

Accuracy_inbox_total=np.zeros(9)
Accuracy_outbox_total=np.zeros(9)
   
#-------------------------------------------------------
#start to plot


plt.figure()


plt.subplots_adjust(left=0.1, right=0.95,bottom=0.12,top=0.95, wspace=0.45, hspace=0)
plt1=plt.subplot(3,3,1)


plt.scatter(mgjh[idx3r0],sio2[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(mgjh[idx2r0],sio2[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
plt.scatter(mgjh[idx1r0],sio2[idx1r0],marker='x',c='deepskyblue',edgecolors='deepskyblue',s=15,linewidth=0.5)



plt.scatter(mgjh[idx3w0],sio2[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)
plt.scatter(mgjh[idx2w0],sio2[idx2w0],marker='s',edgecolors='orangered',facecolor='gold',s=15,linewidth=0.8)
plt.scatter(mgjh[idx1w0],sio2[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)


rect = plt.Rectangle((0.65,42),0.2,15,fill='False',facecolor='none',edgecolor='k')
plt1.add_patch(rect)


#----------------------------------------------------------------------
#determine the accuracy in the box
g1=np.where((sio2>=42) & (sio2<=57) & (mgjh>=0.65) & (mgjh<=0.85))    #right
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

plt.figtext(0.15,0.7,"%.0f%%"  % value1)

#----------------------------------------------------------------------





#determine the accuracy out of the box



#plt.xlabel('MgO(%)',fontsize=12)          
plt.ylabel('Si$\mathregular{O_2}$(wt%)',fontsize=10) 
plt.xticks([])

plt.figtext(0.28,0.9,'(a)',color='black',fontsize=12)

#-----
plt2=plt.subplot(3,3,2)

plt.scatter(mgjh[idx3r0],tio2[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(mgjh[idx2r0],tio2[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
plt.scatter(mgjh[idx1r0],tio2[idx1r0],marker='x',c='deepskyblue',s=15,linewidth=0.5)



plt.scatter(mgjh[idx3w0],tio2[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)
plt.scatter(mgjh[idx2w0],tio2[idx2w0],marker='s',edgecolors='orangered',facecolor='gold',s=15,linewidth=0.8)
plt.scatter(mgjh[idx1w0],tio2[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)



rect = plt.Rectangle((0.65,0),0.2,3,fill='False',facecolor='none',edgecolor='k')
plt2.add_patch(rect)


#plt.xlabel('MgO(%)',fontsize=12)          
plt.ylabel('Ti$\mathregular{O_2}$(wt%)',fontsize=10) 

plt.figtext(0.6,0.9,'(b)',color='black',fontsize=12)
plt.xticks([])

#plt.figtext(0.55,0.85,'S_ID*RA=74%',color='black',fontsize=12)



#----------------------------------------------------------------------
#determine the accuracy in the box
g2=np.where((tio2>=0) & (tio2<=3) & (mgjh>=0.65) & (mgjh<=0.85))    #right
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

plt.figtext(0.5,0.83,"%.0f%%"  % value2)
#----------------------------------------------------------------------






#-----
plt3=plt.subplot(3,3,3)

plt.scatter(mgjh[idx3r0],al2o3[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(mgjh[idx2r0],al2o3[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
plt.scatter(mgjh[idx1r0],al2o3[idx1r0],marker='x',c='deepskyblue',s=15,linewidth=0.5)




plt.scatter(mgjh[idx3w0],al2o3[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)
plt.scatter(mgjh[idx2w0],al2o3[idx2w0],marker='s',edgecolors='orangered',facecolor='gold',s=15,linewidth=0.8)
plt.scatter(mgjh[idx1w0],al2o3[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)


rect = plt.Rectangle((0.65,10),0.2,10,fill='False',facecolor='none',edgecolor='k')
plt3.add_patch(rect)

plt.xticks([])
#plt.xlabel('MgO(%)',fontsize=11)          
plt.ylabel('$\mathregular{Al_2}$$\mathregular{O_3}$(wt%)',fontsize=10) 

plt.figtext(0.91,0.9,'(c)',color='black',fontsize=12)





#----------------------------------------------------------------------
#determine the accuracy in the box
g3=np.where((al2o3>=10) & (al2o3<=20) & (mgjh>=0.65) & (mgjh<=0.85))    #right
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

plt.figtext(0.8,0.7,"%.0f%%"  % value3)
#----------------------------------------------------------------------












#-----
plt4=plt.subplot(3,3,4)

plt.scatter(mgjh[idx3r0],mgo[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)

plt.scatter(mgjh[idx2r0],mgo[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)

plt.scatter(mgjh[idx1r0],mgo[idx1r0],marker='x',c='deepskyblue',s=15,linewidth=0.5)


plt.scatter(mgjh[idx3w0],mgo[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)

plt.scatter(mgjh[idx2w0],mgo[idx2w0],marker='s',edgecolors='orangered',facecolor='gold',s=15,linewidth=0.8)

plt.scatter(mgjh[idx1w0],mgo[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)

plt4.legend(['Mafic R', 'Transitional R', 'Peridotite R','Mafic W','Transitional W','Peridotite W'], loc='upper left',fontsize=4)



rect = plt.Rectangle((0.65,8),0.2,15,fill='False',facecolor='none',edgecolor='k')
plt4.add_patch(rect)

#plt.xlabel('MgO(%)',fontsize=12)          
plt.ylabel('MgO(wt%)',fontsize=10) 
plt.xticks([])

plt.figtext(0.28,0.61,'(d)',color='black',fontsize=12)




#----------------------------------------------------------------------
#determine the accuracy in the box
g4=np.where((mgo>=8) & (mgo<=23) & (mgjh>=0.65) & (mgjh<=0.85))    #right
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

value4=100*Same_out_box

plt.figtext(0.12,0.48,"%.0f%%"  % value4)
#----------------------------------------------------------------------







#-----
plt5=plt.subplot(3,3,5)

plt.scatter(mgjh[idx3r0],feo[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(mgjh[idx2r0],feo[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
plt.scatter(mgjh[idx1r0],feo[idx1r0],marker='x',c='deepskyblue',s=15,linewidth=0.5)


plt.scatter(mgjh[idx3w0],feo[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)
plt.scatter(mgjh[idx2w0],feo[idx2w0],marker='s',edgecolors='orangered',facecolor='gold',s=15,linewidth=0.8)
plt.scatter(mgjh[idx1w0],feo[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)

rect = plt.Rectangle((0.65,5),0.2,8,fill='False',facecolor='none',edgecolor='k')
plt5.add_patch(rect)
plt.xticks([])


#plt.xlabel('MgO(%)',fontsize=12)          
plt.ylabel('FeO(wt%)',fontsize=10) 

plt.figtext(0.6,0.61,'(e)',color='black',fontsize=12)


#----------------------------------------------------------------------
#determine the accuracy in the box
g5=np.where((feo>=5) & (feo<=13) & (mgjh>=0.65) & (mgjh<=0.85))    #right
g50=g5[0]

diff_in_box=difference_total[g50]

N_diff=np.where(diff_in_box==0)  
N_diff0=N_diff[0]
Same_in_box=len(N_diff0)/len(diff_in_box)
print('Number in box fig5:    ',len(diff_in_box))   
print('Accuracy in box fig5:    ',Same_in_box)   

idx=np.where(difference_total==0)  
idx0=idx[0]
N_total_same=len(idx0)
Same_out_box=(N_total_same-len(N_diff0))/(915-len(diff_in_box))

print('Number out box fig5:    ',915-len(diff_in_box))   
print('Accuracy out box fig5:    ',Same_out_box)   

print('----------------------------------------------------------')   

N_inbox_total[4]=len(diff_in_box)
N_outbox_total[4]=915-len(diff_in_box)

Accuracy_inbox_total[4]=Same_in_box
Accuracy_outbox_total[4]=Same_out_box

value5=100*Same_out_box

plt.figtext(0.5,0.41,"%.0f%%"  % value5)

#----------------------------------------------------------------------












#-----
plt6=plt.subplot(3,3,6)

plt.scatter(mgjh[idx3r0],mno[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(mgjh[idx2r0],mno[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
plt.scatter(mgjh[idx1r0],mno[idx1r0],marker='x',c='deepskyblue',s=15,linewidth=0.5)


plt.scatter(mgjh[idx3w0],mno[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)
plt.scatter(mgjh[idx2w0],mno[idx2w0],marker='s',edgecolors='orangered',facecolor='gold',s=15,linewidth=0.8)
plt.scatter(mgjh[idx1w0],mno[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)


rect = plt.Rectangle((0.65,0),0.2,0.25,fill='False',facecolor='none',edgecolor='k')
plt6.add_patch(rect)

#plt.xlabel('MgO(%)',fontsize=12)          
plt.ylabel('MnO(wt%)',fontsize=10) 
plt.xticks([])

plt.figtext(0.91,0.61,'(f)',color='black',fontsize=12)




#----------------------------------------------------------------------
#determine the accuracy in the box
g6=np.where((mno>=0) & (mno<=0.25) & (mgjh>=0.65) & (mgjh<=0.85))    #right
g60=g6[0]

diff_in_box=difference_total[g60]
N_diff=np.where(diff_in_box==0)  
N_diff0=N_diff[0]
Same_in_box=len(N_diff0)/len(diff_in_box)
print('Number in box fig6:    ',len(diff_in_box))   
print('Accuracy in box fig6:    ',Same_in_box)   

idx=np.where(difference_total==0)  
idx0=idx[0]
N_total_same=len(idx0)
Same_out_box=(N_total_same-len(N_diff0))/(915-len(diff_in_box))

print('Number out box fig6:    ',915-len(diff_in_box))   
print('Accuracy out box fig6:    ',Same_out_box)   

print('----------------------------------------------------------')   

N_inbox_total[5]=len(diff_in_box)
N_outbox_total[5]=915-len(diff_in_box)

Accuracy_inbox_total[5]=Same_in_box
Accuracy_outbox_total[5]=Same_out_box


value6=100*Same_out_box

plt.figtext(0.8,0.43,"%.0f%%"  % value6)
#----------------------------------------------------------------------














#-----
plt7=plt.subplot(3,3,7)

plt.scatter(mgjh[idx3r0],cao[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(mgjh[idx2r0],cao[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)
plt.scatter(mgjh[idx1r0],cao[idx1r0],marker='x',c='deepskyblue',s=15,linewidth=0.5)

plt.scatter(mgjh[idx3w0],cao[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)
plt.scatter(mgjh[idx2w0],cao[idx2w0],marker='s',edgecolors='orangered',facecolor='gold',s=15,linewidth=0.8)
plt.scatter(mgjh[idx1w0],cao[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)

rect = plt.Rectangle((0.65,6),0.2,9,fill='False',facecolor='none',edgecolor='k')
plt7.add_patch(rect)


plt.xlabel('MgO#',fontsize=10)          
plt.ylabel('CaO(wt%)',fontsize=10) 

plt.figtext(0.28,0.34,'(g)',color='black',fontsize=12)





#----------------------------------------------------------------------
#determine the accuracy in the box
g7=np.where((cao>=6) & (cao<=15) & (mgjh>=0.65) & (mgjh<=0.85))    #right
g70=g7[0]

diff_in_box=difference_total[g70]
N_diff=np.where(diff_in_box==0)  
N_diff0=N_diff[0]
Same_in_box=len(N_diff0)/len(diff_in_box)
print('Number in box fig7:    ',len(diff_in_box))   
print('Accuracy in box fig7:    ',Same_in_box)   

idx=np.where(difference_total==0)  
idx0=idx[0]
N_total_same=len(idx0)
Same_out_box=(N_total_same-len(N_diff0))/(915-len(diff_in_box))

print('Number out box fig7:    ',915-len(diff_in_box))   
print('Accuracy out box fig7:    ',Same_out_box)   

print('----------------------------------------------------------')   

N_inbox_total[6]=len(diff_in_box)
N_outbox_total[6]=915-len(diff_in_box)

Accuracy_inbox_total[6]=Same_in_box
Accuracy_outbox_total[6]=Same_out_box


value7=100*Same_out_box

plt.figtext(0.12,0.27,"%.0f%%"  % value7)
#----------------------------------------------------------------------













#-----
plt8=plt.subplot(3,3,8)

plt.scatter(mgjh[idx3r0],na2o[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)

plt.scatter(mgjh[idx2r0],na2o[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)

plt.scatter(mgjh[idx1r0],na2o[idx1r0],marker='x',c='deepskyblue',s=15,linewidth=0.5)





plt.scatter(mgjh[idx3w0],na2o[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)

plt.scatter(mgjh[idx2w0],na2o[idx2w0],marker='s',edgecolors='orangered',facecolor='gold',s=15,linewidth=0.8)

plt.scatter(mgjh[idx1w0],na2o[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)


rect = plt.Rectangle((0.65,0.05),0.2,3,fill='False',facecolor='none',edgecolor='k')
plt8.add_patch(rect)

plt.xlabel('MgO#',fontsize=10)          
plt.ylabel('$\mathregular{Na_2}$O(wt%)',fontsize=10) 

plt.figtext(0.595,0.34,'(h)',color='black',fontsize=12)




#----------------------------------------------------------------------
#determine the accuracy in the box
g8=np.where((na2o>=0.05) & (na2o<=3) & (mgjh>=0.65) & (mgjh<=0.85))    #right
g80=g8[0]

diff_in_box=difference_total[g80]
N_diff=np.where(diff_in_box==0)  
N_diff0=N_diff[0]
Same_in_box=len(N_diff0)/len(diff_in_box)
print('Number in box fig8:    ',len(diff_in_box))   
print('Accuracy in box fig8:    ',Same_in_box)   

idx=np.where(difference_total==0)  
idx0=idx[0]
N_total_same=len(idx0)
Same_out_box=(N_total_same-len(N_diff0))/(915-len(diff_in_box))

print('Number out box fig8:    ',915-len(diff_in_box))   
print('Accuracy out box fig8:    ',Same_out_box)   

print('----------------------------------------------------------')   

N_inbox_total[7]=len(diff_in_box)
N_outbox_total[7]=915-len(diff_in_box)

Accuracy_inbox_total[7]=Same_in_box
Accuracy_outbox_total[7]=Same_out_box


value8=100*Same_out_box

plt.figtext(0.46,0.13,"%.0f%%"  % value8)
#----------------------------------------------------------------------










#-----
plt9=plt.subplot(3,3,9)


plt.scatter(mgjh[idx3r0],k2o[idx3r0],marker='^',c='',edgecolors='0.5',s=15,linewidth=0.5)
plt.scatter(mgjh[idx2r0],k2o[idx2r0],marker='s',c='limegreen',edgecolors='k',s=15,linewidth=0.5)

plt.scatter(mgjh[idx1r0],k2o[idx1r0],marker='x',c='deepskyblue',s=15,linewidth=0.5)



plt.scatter(mgjh[idx3w0],k2o[idx3w0],marker='^',c='0.7',edgecolors='orangered',s=15,linewidth=0.8)

plt.scatter(mgjh[idx2w0],k2o[idx2w0],marker='s',edgecolors='orangered',facecolor='gold',s=15,linewidth=0.8)

plt.scatter(mgjh[idx1w0],k2o[idx1w0],marker='x',c='orangered',s=15,linewidth=0.8)


rect = plt.Rectangle((0.65,0),0.2,4,fill='False',facecolor='none',edgecolor='k')
plt9.add_patch(rect)

plt.xlabel('MgO#',fontsize=10)          
plt.ylabel('$\mathregular{K_2}$O(wt%)',fontsize=10) 

plt.figtext(0.91,0.34,'(i)',color='black',fontsize=12)



#----------------------------------------------------------------------
#determine the accuracy in the box
g9=np.where((k2o>=0) & (k2o<=4) & (mgjh>=0.65) & (mgjh<=0.85))    #right
g90=g9[0]

diff_in_box=difference_total[g90]
N_diff=np.where(diff_in_box==0)  
N_diff0=N_diff[0]
Same_in_box=len(N_diff0)/len(diff_in_box)
print('Number in box fig9:    ',len(diff_in_box))   
print('Accuracy in box fig9:    ',Same_in_box)   

idx=np.where(difference_total==0)  
idx0=idx[0]
N_total_same=len(idx0)
Same_out_box=(N_total_same-len(N_diff0))/(915-len(diff_in_box))

print('Number out box fig9:    ',915-len(diff_in_box))   
print('Accuracy out box fig9:    ',Same_out_box)   

print('----------------------------------------------------------')   

N_inbox_total[8]=len(diff_in_box)
N_outbox_total[8]=915-len(diff_in_box)

Accuracy_inbox_total[8]=Same_in_box
Accuracy_outbox_total[8]=Same_out_box

value9=100*Same_out_box

plt.figtext(0.78,0.25,"%.0f%%"  % value9)


#----------------------------------------------------------------------



#----save fig
plt.savefig('resultcompare_checkdata2',dpi=300)

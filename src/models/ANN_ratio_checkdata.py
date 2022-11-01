#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 19:24:37 2021

@author: lilucheng
"""




import pandas as pd
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from numpy import *
import seaborn as sns
from sklearn.metrics import confusion_matrix


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

#-------------------------------------------------------
#ratio
#FCKANTMS     feo/cao    k2o/al2o3  tio2/na2o      na2o/k2o   na2o/tio2  mgo/sio2
#feo cao k2o al2o3  tio2 na2o  mgo sio2
   

feocao=feo/cao
k2oal2o3=k2o/al2o3
tio2na2o=tio2/na2o
na2ok2o=na2o/k2o
na2otio2=na2o/tio2
mgosio2=mgo/sio2


newX=np.zeros((915,6))

feocao=feocao.reshape(1,-1)
k2oal2o3=k2oal2o3.reshape(1,-1)
tio2na2o=tio2na2o.reshape(1,-1)
na2ok2o=na2ok2o.reshape(1,-1)
na2otio2=na2otio2.reshape(1,-1)
mgosio2=mgosio2.reshape(1,-1)






newX[:,0]=feocao
newX[:,1]=k2oal2o3
newX[:,2]=tio2na2o
newX[:,3]=na2ok2o
newX[:,4]=na2otio2
newX[:,5]=mgosio2

newy=y



X_train, X_test, y_train, y_test = train_test_split(newX, newy, train_size=0.8, random_state = 0)


clf = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 10), random_state=1)


clf = clf.fit(X_train, y_train)




accuracy_ANN = clf.score(X_test, y_test)

print('Accuracy Neural network:', accuracy_ANN)


#visilize
# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
accuracy_ANNtrain = clf.score(X_train, y_train)

print('Accuracy Neural network of train:', accuracy_ANNtrain)

#figsize=(10, 10)
fig, ax = plt.subplots()

plot_confusion_matrix(clf, X_train, y_train,normalize='true',ax=ax)  



fig.savefig('ann_ratio_checkdata',dpi=300)




   
accuracy_ratio_all=clf.score(newX,newy)
print('Accuracy ratio of all data :', accuracy_ratio_all)


plot_confusion_matrix(clf, newX, newy,normalize='true')  
#plt.title('Accuracy for overall data:' accuracy_ratio_all)

plt.xlabel('Predicted label\naccuracy={:0.4f}'.format(accuracy_ratio_all))

y_predict=clf.predict(newX)

#---------------------------------------------------
#prepare for natural data

labels = ['Peridotitic', 'Transitional','Mafic']
cm = confusion_matrix(newy,y_predict)


ax= plt.subplot()
plt.subplots_adjust(left=0.17, right=0.9,bottom=0.18,top=0.9, wspace=0, hspace=0)

sns.heatmap(cm, annot=True, fmt='g', ax=ax,cmap="YlGnBu");  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted');ax.set_ylabel('True'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Peridotitic', 'Transitional','Mafic']);
ax.yaxis.set_ticklabels(['Peridotitic','Transitional', 'Mafic']);
plt.xlabel('Predict\nAccuracy for overall data={:0.0f}%'.format(accuracy_ratio_all*100))

plt.savefig('confusion_for_ratio_ann',dpi=300)



#make the confusion matrix
y_all=clf.predict(newX)

cm=confusion_matrix(newy,y_all)

#here we calculate P persion=tp/(tp+fp),R recall=tp/(tp+fn),f1  =(P+R)/2*P*R
a=len(cm)
P=np.zeros(3)
R=np.zeros(3)
F1=np.zeros(3)
for i in range(0,a):
    P[i]=cm[i][i]/sum(cm[:,i])
    R[i]=cm[i][i]/sum(cm[i,:])
    F1[i]=(2*P[i]*R[i])/(P[i]+R[i])

P_mean=mean(P)
R_mean=mean(R)
F1_mean=mean(F1)

print(P)
print(R)
print(F1)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:57:01 2021

@author: lilucheng
"""

#ann with fine data we filter out several data which 0 pressusr


import pandas as pd
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from numpy import *
from sklearn.metrics import mean_squared_error 
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score




import sys
sys.path.append(".")
from cf_matrix import make_confusion_matrix
#import cf_matrix
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

#default
#clf = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 20), random_state=1)


clf = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 20), random_state=1)
#clf = MLPClassifier(activation='relu',solver='sgd', alpha=1e-5,hidden_layer_sizes=(20, 20), random_state=1)
#clf = MLPClassifier(activation='relu',solver='adam', alpha=1e-5,hidden_layer_sizes=(20, 20), random_state=1)


clf = clf.fit(X_train, y_train)




accuracy_ANN = clf.score(X_test, y_test)

print('Accuracy Neural network test:', accuracy_ANN)


# =============================================================================
# #k-cross
# accuracies = cross_val_score(clf, X = X_train, y = y_train, cv = 10)
# mean_ac = accuracies.mean()
# variance_ac = accuracies.std()
# 
# print('Accuracy Neural network k-cross mean:', mean_ac)
# print('Accuracy Neural network k-cross std:', variance_ac)
# 
# =============================================================================


#visilize
# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
accuracy_ANNtrain = clf.score(X_train, y_train)

y_train_predict=clf.predict(X_train)



print('Accuracy Neural network of train:', accuracy_ANNtrain)



#-------------------------------------------------------------

y_test_predict=clf.predict(X_test)


#-------------------------------------------------------------

y_result=clf.predict(newX) 
y_result=y_result.reshape(-1,1) 
difference_total=np.zeros((915,1))
difference_total=newy-y_result
idx=np.where(difference_total==0)  
idx0=idx[0]
Same_alldata=len(idx0)/len(y_result)
print('Accuracy Neural network of all data:', Same_alldata)



plt.figure()

plot_confusion_matrix(clf, X_test, y_test,normalize='true')  
#plot_confusion_matrix(clf, newX, newy,normalize='true')  


#plt.xticks('Mafic','Transitional','Peridotite')

plt.xticks([0, 1, 2], ['Peridotite', 'Transitional', 'Mafic'])  # Se
plt.yticks([0, 1, 2], ['Peridotite', 'Transitional', 'Mafic'])  # Se

#plt.yticks('Mafic','Transitional','Peridotite')
#plt.show()
#----save fig






plot_confusion_matrix(clf, X_train, y_train,normalize='true')  
#plot_confusion_matrix(clf, newX, newy,normalize='true')  


#plt.xticks('Mafic','Transitional','Peridotite')

plt.xticks([0, 1, 2], ['Peridotite', 'Transitional', 'Mafic'])  # Se
plt.yticks([0, 1, 2], ['Peridotite', 'Transitional', 'Mafic'])  # Se

#plt.yticks('Mafic','Transitional','Peridotite')
#plt.show()
#----save fig


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


#----------------------------------------
#plot 

plot_confusion_matrix(clf, newX, newy)  


plt.xticks([0, 1, 2], ['Peridotite', 'Transitional', 'Mafic'])  # Se
plt.yticks([0, 1, 2], ['Peridotite', 'Transitional', 'Mafic'])  # Se




stats_text = "Precision ={:0.2f},  {:0.2f},  {:0.2f}\n   Recall   ={:0.2f},  {:0.2f},  {:0.2f}\n      F1      ={:0.2f},  {:0.2f},  {:0.2f}".format(P[0],P[1],P[2],R[0],R[1],
                                                                                       R[1],F1[0],F1[1],F1[2])
plt.xlabel('Predict\n {}'.format(stats_text))
    
    
plt.ylabel('True')
plt.title('Sample size:313,103,499')

#plt.savefig('313-103-499confusion.png',dpi=300)

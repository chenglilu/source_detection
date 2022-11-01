#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:02:04 2021

@author: lilucheng
"""

#read the data and make some change using pandas


import pandas as pd



df = pd.read_excel('samplesofLee.xlsx',header = 0,index_col=0)
df.isnull().any()
#print(df.isnull().any())
df.fillna(value=0, inplace=True)
#print(df.isnull().any())
#print(df.columns)
df['FeOt']=df['FeO']+0.9*df['Fe2O3']


#take part of table as the input data based on the modeling requirements
df_input=df[['SiO2','TiO2','Al2O3','Cr2O3','FeOt','MnO','MgO','CaO','Na2O','K2O','T','Gpa.7']]


df_input['P/T']=df_input['Gpa.7']*1000/df_input['T']

print(df_input.columns)



df_input2=df_input[df_input['P/T']<=5]

print(df_input2.columns)
print(len(df_input2))

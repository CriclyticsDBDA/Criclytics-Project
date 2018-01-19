# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 03:04:03 2018

@author: Chinmay
"""


import pandas as pd
import numpy as np
import ast

#Import dataset
df = pd.read_csv("final_all_2.0.csv")

teams = ['Australia' , 'New Zealand' , 'India' , 'Zimbabwe' , 'Bangladesh' , 'South Africa' , 'England' 
         , 'Sri Lanka' , 'Pakistan' , 'West Indies' , 'Ireland']


df['info.teams'] = df['info.teams'].str.strip().apply(ast.literal_eval)

ndf = df['info.teams'].apply(pd.Series)

#idx = ndf[ndf.isin(['India' , 'Sri Lanka']).all(1)].index

idx = ndf[(ndf[0]=='India') | (ndf[1]=='India')].index

type(idx)



df = df.loc[idx]

list(df)

df_try = df.copy()
df = df_try

match_index = df['index_all'].unique()


batsman_data = pd.DataFrame()
for mindex in match_index:
    df_match = df[(df['index_all']==mindex)].copy()
    
    if df_match[(df['batsman']=='RG Sharma')].empty:
        continue
    runs = 0
    balls_faced = 0
    kohli_data = []
    for indexs,match_details in df_match.iterrows():
        batsman = df_match.loc[indexs,'batsman']        
        if (df_match.loc[indexs,'batsman'] == 'RG Sharma'):
            runs = runs + df_match.loc[indexs,'runs.batsman']
            balls_faced = balls_faced + 1
    print(runs)
    kohli_data.append(df_match.loc[indexs,'info.match_type'])
    kohli_data.append(df_match.loc[indexs,'info.dates'])
    kohli_data.append(df_match.loc[indexs,'info.neutral_venue'])
    kohli_data.append(df_match.loc[indexs,'info.teams'])
    #kohli_data.append(mindex)
    kohli_data.append(runs)
    kohli_data.append(balls_faced)
    batsman_data = batsman_data.append(pd.Series(kohli_data ), ignore_index=True)
    print(mindex)
batsman_data.columns = ['match_type' , 'date' , 'neutral' , 'teams' , 'runs' , 'balls_faced']


df_match[(df['batsman']=='RG Sharma')]['runs.batsman'].sum()

avg = df[(df['info.match_type']=='ODI') & (df['batsman']=='RG Sharma')]['runs.batsman'].sum() / df[(df['info.match_type']=='ODI') & (df['batsman']=='RG Sharma')]['wicket.kind'].count()

from sklearn.naive_bayes import GaussianNB
import numpy as np

d = zip(batsman_data['runs'],batsman_data['balls_faced'])
x = np.array(batsman_data['runs'],batsman_data['balls_faced'])


A = batsman_data.loc[:, 'balls_faced'].values
B = batsman_data.loc[:, 'runs'].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(A, B, test_size = 0.05, random_state = 0)


x = []
for f,b in zip(X_train,X_train):
    x.append([f,b])



#x = x.reshape(1, -1)

Y = np.array(y_train)


model = GaussianNB()

model.fit(x, Y)

pred_x = []
for f,b in zip(X_test,X_test):
    pred_x.append([f,b])
#Predict Output 
predicted= model.predict(pred_x)
print(predicted)
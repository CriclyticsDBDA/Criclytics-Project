# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 00:37:31 2018

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

#idx = ndf[ndf.isin(['India' , 'Australia']).all(1)].index

idx = ndf[(ndf[0]=='India') | (ndf[1]=='India')].index

type(idx)



df = df.loc[idx]

list(df)

df_try = df.copy()
df = df_try

match_index = df['index_all'].unique()


batsman_all_data = pd.read_csv("all_batsmans_data.csv")


batsman_data = pd.DataFrame()
for mindex in match_index:
    df_match = df[(df['index_all']==mindex)].copy()
    
    if df_match[(df['batsman']=='RG Sharma')].empty:
        continue
    runs = 0
    balls_faced = 0
    kohli_data = []
    for indexs,match_details in df_match.iterrows():
        ls = list(df_match.loc[indexs , 'info.teams'])
        if ls[0] == 'India':
            opposition = ls[1]
        else:
            opposition = ls[0]
        batsman = df_match.loc[indexs,'batsman']        
        if (df_match.loc[indexs,'batsman'] == 'RG Sharma'):
            runs = runs + df_match.loc[indexs,'runs.batsman']
            balls_faced = balls_faced + 1
    print(runs)
    kohli_data.append(df_match.loc[indexs,'info.match_type'])
    kohli_data.append(df_match.loc[indexs,'info.dates'])
    kohli_data.append(df_match.loc[indexs,'info.neutral_venue'])
    kohli_data.append(opposition)
    #kohli_data.append(mindex)
    kohli_data.append(runs)
    kohli_data.append(balls_faced)
    kohli_data.append(df_match.loc[indexs,'info.venue'])
    batsman_data = batsman_data.append(pd.Series(kohli_data ), ignore_index=True)
    print(mindex)
batsman_data.columns = ['match_type' , 'date' , 'neutral' , 'against' , 'runs' , 'balls_faced' , 'venue']


df_match[(df['batsman']=='RG Sharma')]['runs.batsman'].sum()

avg = df[(df['info.match_type']=='ODI') & (df['batsman']=='RG Sharma')]['runs.batsman'].sum() / df[(df['info.match_type']=='ODI') & (df['batsman']=='RG Sharma')]['wicket.kind'].count()


from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import numpy as np

d = zip(batsman_data['runs'],batsman_data['balls_faced'])
x = np.array(batsman_data['runs'],batsman_data['balls_faced'])

batsman_data['team_encoded'] = batsman_data['against'].astype('category').cat.codes

batsman_data['venue_encoded'] = batsman_data['venue'].astype('category').cat.codes


#batsman_data['teams'].astype('category').cat.codes

A = batsman_data.loc[:, ['balls_faced' , 'team_encoded' , 'venue_encoded']].values
B = batsman_data.loc[:, 'runs'].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(A, B, test_size = 0.05, random_state = 0)


#x = []
#for f,b in zip(X_train,X_train):
#    x.append([f,b])



#x = x.reshape(1, -1)

Y = np.array(y_train)


model = MultinomialNB()

model.fit(X_train, y_train)

pred_x = []
for f,b in zip(X_test,X_test):
    pred_x.append([f,b])
#Predict Output 
predicted= model.predict(X_test)
print(predicted)



import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set.min() - 1, stop = X_set.max() + 1, step = 10),
                     np.arange(start = X_set.min() - 1, stop = X_set.max() + 1, step = 10))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Balls')
plt.ylabel('Estimated Runs')
plt.legend()
plt.show()


from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set.min() - 1, stop = X_set.max() + 1, step = 10),
                     np.arange(start = X_set.min() - 1, stop = X_set.max() + 1, step = 10))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Balls')
plt.ylabel('Estimated Runs')
plt.legend()
plt.show()
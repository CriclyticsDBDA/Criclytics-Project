# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 03:04:03 2018

@author: Chinmay
"""


import pandas as pd
import numpy as np
import ast

#Import dataset
df = pd.read_csv("final_all_3.0.csv")

teams = ['Australia' , 'New Zealand' , 'India' , 'Zimbabwe' , 'Bangladesh' , 'South Africa' , 'England' 
         , 'Sri Lanka' , 'Pakistan' , 'West Indies' , 'Ireland']


df['info.teams'] = df['info.teams'].str.strip().apply(ast.literal_eval)

ndf = df['info.teams'].apply(pd.Series)

#idx = ndf[ndf.isin(['Australia' , 'England']).all(1)].index
#ndf.to_csv("ndf.csv")

all_batsmans_data = pd.DataFrame()

for team in teams:
    
    print(team)
    
    idx = ndf[(ndf[0]==team) | (ndf[1]==team)].index
    
    #type(idx)
    
    df_team = df.loc[idx]

    #list(df)
    
    
    batsmans = df_team[(df_team['team']==team)]['batsman'].unique()
    
    
    df_try = df.copy()
    #df = df_try
    
    match_index = df_team['index_all'].unique()
    
    i = 0
    
    for batsman in batsmans:
        i = i + 1
        print(i)
        print(batsman)
        batsman_data = pd.DataFrame()
        df_batsman_bowler_record_total = pd.DataFrame( columns = ['match_id' , 'bowler' , 'runs_scored' , 'balls'])
    
        for mindex in match_index:
            df_match = df_team[(df_team['index_all']==mindex)].copy()
            
            ls = list(df_match.iloc[1]['info.teams'])
            if ls[0] == team:
                opposition = ls[1]
            else:
                opposition = ls[0]
            
            bowlers = df_match[(df_match['team']==team)]['bowler'].unique()

            if df_match[(df_match['batsman']==batsman)].empty:
                continue
            runs = 0
            balls_faced = 0
            bowler_no = 0
            batsman_list = []
            df_batsman_bowler_record = pd.DataFrame( columns = ['match_id' ,'bowler' , 'runs_scored' , 'balls'])
            
            for indexs,match_details in df_match.iterrows():
                ls = list(df_match.loc[indexs , 'info.teams'])
                if ls[0] == team:
                    home = 0
                    opposition = ls[1]
                else:
                    home = 1
                    opposition = ls[0]
                #batsman = df_match.loc[indexs,'batsman']        
                if (df_match.loc[indexs,'batsman'] == batsman):
                    runs = runs + df_match.loc[indexs,'runs.batsman']
                    balls_faced = balls_faced + 1
                    
                    runs_against_the_bowler = 0
                    balls_against_the_bowler = 0
                    for bowler in bowlers:
                        if df_batsman_bowler_record[(df_batsman_bowler_record['bowler']==bowler)].empty:
                            
                            bowler_no = bowler_no + 1
                            bowler_batsman = []
                            bowler_batsman.append(mindex)
                            bowler_batsman.append(bowler)
                            bowler_batsman.append(df_match[(df_match['batsman']==batsman) & (df_match['bowler']==bowler)]['runs.batsman'].sum())
                            bowler_batsman.append(df_match[(df_match['batsman']==batsman) & (df_match['bowler']==bowler)]['runs.batsman'].count())
                            df_batsman_bowler_record.loc[bowler_no] = bowler_batsman
            #print(runs)
            df_batsman_bowler_record_total = df_batsman_bowler_record_total.append(df_batsman_bowler_record)
            batsman_list.append(mindex)
            batsman_list.append(team)
            batsman_list.append(batsman)
            batsman_list.append(df_match.loc[indexs,'info.match_type'])
            batsman_list.append(df_match.loc[indexs,'info.dates'])
            batsman_list.append(df_match.loc[indexs,'info.neutral_venue'])
            batsman_list.append(opposition)
            batsman_list.append(home)
            batsman_list.append(df_match.loc[indexs,'info.venue'])
            #kohli_data.append(mindex)
            batsman_list.append(runs)
            batsman_list.append(balls_faced)
            batsman_data = batsman_data.append(pd.Series(batsman_list ), ignore_index=True)
            #print(mindex)
            #break
        batsman_data.columns = ['match_id', 'for team' ,'name', 'match_type' , 'date' , 'neutral' , 'against' ,'home_away', 'venue', 'runs' , 'balls_faced']
        batsman_data_try = pd.merge(batsman_data, df_batsman_bowler_record_total, how='left', on='match_id')
            #all_batsmans_data.columns = ['name', 'match_type' , 'date' , 'neutral' , 'teams' , 'runs' , 'balls_faced']
        all_batsmans_data = all_batsmans_data.append(batsman_data_try)
        
all_batsmans_data.to_csv("all_batsmans_data_2.0.csv")

df_match[(df_team['batsman']=='RG Sharma')]['runs.batsman'].sum()

avg = df_team[(df_team['info.match_type']=='ODI') & (df_team['batsman']=='RG Sharma')]['runs.batsman'].sum() / df_team[(df_team['info.match_type']=='ODI') & (df_team['batsman']=='RG Sharma')]['wicket.kind'].count()

#!conda install skmultilearn.problem_transform
#from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB
import numpy as np

batsman_data = all_batsmans_data[(all_batsmans_data['name'] == 'DA Warner')  & (all_batsmans_data['match_type'] == 'ODI')]

batsman_data_team = batsman_data[(batsman_data['against'] == 'England') & (batsman_data['home_away'] == 0)]

avg_balls = batsman_data_team['balls_faced'].sum() / batsman_data_team['balls_faced'].count()

#d = zip(batsman_data['runs'],batsman_data['balls_faced'])
#x = np.array(batsman_data['runs'],batsman_data['balls_faced'])

batsman_data['against_encoded'] = batsman_data['against'].astype('category').cat.codes

batsman_data['venue_encoded'] = batsman_data['venue'].astype('category').cat.codes
A = batsman_data.loc[:, ['balls_faced' , 'against_encoded']].values
B = batsman_data.loc[:, 'runs'].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(A, B, test_size = 0.05, random_state = 0)


#x = []
#for f,b in zip(X_train,X_train):
#    x.append([f,b])



#x = x.reshape(1, -1)

Y = np.array(y_train)


model = GaussianNB()

model.fit(X_train, y_train)

#pred_x = []
#for f,b in zip(X_test,X_test):
#    pred_x.append([f,b])
#Predict Output 
predicted= model.predict(X_test)
print(predicted)


from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)



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
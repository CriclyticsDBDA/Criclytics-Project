# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 12:59:08 2018

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

#idx = ndf[ndf.isin(['India' , 'Sri Lanka']).all(1)].index

idx = ndf[(ndf[0]=='India') | (ndf[1]=='India')].index

type(idx)

df_team = df.loc[idx]



batsman_all_data = pd.read_csv("all_batsmans_data.csv")


match_type = 'ODI'

team1 = 'India'
team2 = 'Australia'


batsmans = batsman_all_data[(batsman_all_data['for team'] == team2) & (batsman_all_data['against'] == team1) & (batsman_all_data['match_type'] == match_type)]['name'].unique()


team_1 = [ 'AM Rahane' , 'RG Sharma' , 'V Kohli' , 'MK Pandey', 'KD Jadhav', 'MS Dhoni' , 'HH Pandya'  , 'B Kumar' , 'JJ Bumrah' , 'Kuldeep Yadav' , 'YZ Chahal']

team_2 = ['H Cartwright' , 'DA Warner' , 'SPD Smith' , 'T Head' , 'GJ Maxwell' , 'M Stoinis' , 'MS Wade' , 'A Agar' , 'P Cummins' , 'NM Coulter-Nile' , 'K Richardson']

bowlers_1 = ['B Kumar' , 'JJ Bumrah' , 'Kuldeep Yadav' , 'HH Pandya' , 'YZ Chahal']

bowlers_2 = ['A Agar' , 'P Cummins' , 'NM Coulter-Nile' , 'KW Richardson', 'M Stoinis', 'T Head']


innings_1_score = 0
total_balls = 0

from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.cross_validation import train_test_split

i = 0
for batsman in team_1:
    i = i + 1
    if i >2:
        None#break
    
    
    
    batsman_data = batsman_all_data[(batsman_all_data['name'] == batsman)  & (batsman_all_data['match_type'] == match_type)]
    
    batsman_data_team = batsman_data[(batsman_data['against'] == team2)]
    
    if batsman_data_team.empty:
        continue
    
    batsman_data_team['bowler_encoded'] = batsman_data_team['bowler'].astype('category').cat.codes
    
    A1 = batsman_data_team.loc[:, ['balls' , 'bowler_encoded']].values
    B1 = batsman_data_team.loc[:, 'runs_scored'].values
    
    X1_train, X1_test, y1_train, y1_test = train_test_split(A1, B1, test_size = 0.01, random_state = 0)
    
    
    Y1 = np.array(y1_train)
    
    
    model1 = GaussianNB()
    
    model1.fit(X1_train, y1_train)
    
    test1 = []
    
    for bowler in bowlers_2:
        
        bowl = []
        bowlers =  batsman_data_team['bowler'].unique()
        if batsman_data_team[(batsman_data_team['bowler'] == bowler)]['bowler_encoded'].empty:
            continue
        
        encoded_bowler = batsman_data_team[(batsman_data_team['bowler'] == bowler)]['bowler_encoded'].iloc[0]
        
        if batsman_data_team[(batsman_data_team['bowler_encoded'] == encoded_bowler) & (batsman_data_team['balls'] != 0)]['balls'].count() == 0:
            continue
        
        avg = batsman_data_team[(batsman_data_team['bowler_encoded'] == encoded_bowler) & (batsman_data_team['balls'] != 0)]['balls'].sum()/batsman_data_team[(batsman_data_team['bowler_encoded'] == encoded_bowler) & (batsman_data_team['balls'] != 0)]['balls'].count()

        bowl.append(avg)
        bowl.append(encoded_bowler)
        
        total_balls = total_balls + avg
        test1.append(bowl)
        
        if total_balls > 300 :
            break
         
    #if not test:        
    idx = ndf[(ndf[0]==team1) | (ndf[1]==team1)].index
    df_team = df.loc[idx]
    match_index = df_team['index_all'].unique()
    batsman_data = pd.DataFrame()
    for mindex in match_index:
        
        df_match = df_team[(df_team['index_all']==mindex)].copy()
        
        if df_match[(df_team['batsman']==batsman)].empty:
            continue
        
        runs = 0
        balls_faced = 0
        bats_data = []
        for indexs,match_details in df_match.iterrows():
            ls = list(df_match.loc[indexs , 'info.teams'])
            if ls[0] == team1:
                home = 0
                opposition = ls[1]
            else:
                home = 1
                opposition = ls[0]
        #print(opposition)
            #batsman = df_match.loc[indexs,'batsman']        
            if (df_match.loc[indexs,'batsman'] == batsman):
                runs = runs + df_match.loc[indexs,'runs.batsman']
                balls_faced = balls_faced + 1
        #print(runs)
        bats_data.append(df_match.loc[indexs,'info.match_type'])
        bats_data.append(df_match.loc[indexs,'info.dates'])
        bats_data.append(df_match.loc[indexs,'info.neutral_venue'])
        bats_data.append(home)
        bats_data.append(opposition)
        #kohli_data.append(mindex)
        bats_data.append(runs)
        bats_data.append(balls_faced)
        bats_data.append(df_match.loc[indexs,'info.venue'])
        batsman_data = batsman_data.append(pd.Series(bats_data), ignore_index=True)
        
        #print(mindex)
    batsman_data.columns = ['match_type' , 'date' , 'neutral' ,'home_away', 'against' , 'runs' , 'balls_faced' , 'venue']
    
    

    #batsman_data = batsman_data[(batsman_data['match_type'] == match_type) & (batsman_data['home_away'] == home)]
    batsman_data = batsman_data[(batsman_data['match_type'] == match_type)]
    batsman_data['team_encoded'] = batsman_data['against'].astype('category').cat.codes
    team =  batsman_data['against'].unique()
    
    A = batsman_data.loc[:, ['balls_faced' , 'team_encoded']].values
    B = batsman_data.loc[:, 'runs'].values
    
    X_train, X_test, y_train, y_test = train_test_split(A, B, test_size = 0.01, random_state = 0)

    if batsman_data[(batsman_data['against'] == team2)]['team_encoded'].empty:
        continue
    
    encoded_team = batsman_data[(batsman_data['against'] == team2)]['team_encoded'].iloc[0]
    avg = batsman_data[(batsman_data['team_encoded'] == encoded_team) & (batsman_data['balls_faced'] != 0)]['balls_faced'].sum()/batsman_data[(batsman_data['team_encoded'] == encoded_team) & (batsman_data['balls_faced'] != 0)]['balls_faced'].count()
    
    Y = np.array(y_train)
    model = GaussianNB()
    
    model.fit(X_train, y_train)
    
    test = []
    team = []
    
    team.append(avg)
    team.append(encoded_team)
    test.append(team)
    predicted1= model1.predict(test1)
    print(batsman)
    print(predicted1)
    predicted= model.predict(test)
    print(batsman)
    print(predicted)
    predicted = list(predicted)
    innings_1_score = innings_1_score + sum(predicted)
    
print(innings_1_score)
print(total_balls)


# 2nd Innings
    
innings_2_score = 0
total_balls= 0
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.cross_validation import train_test_split

for batsman in team_2:

    
    batsman_data = batsman_all_data[(batsman_all_data['name'] == batsman)  & (batsman_all_data['match_type'] == match_type)]
    
    batsman_data_team = batsman_data[(batsman_data['against'] == team1)]
    
    if batsman_data_team.empty:
        continue
    
    batsman_data_team['bowler_encoded'] = batsman_data_team['bowler'].astype('category').cat.codes
    
    A = batsman_data_team.loc[:, ['balls' , 'bowler_encoded']].values
    B = batsman_data_team.loc[:, 'runs_scored'].values
    
    X_train, X_test, y_train, y_test = train_test_split(A, B, test_size = 0.01, random_state = 0)
    
    
    Y = np.array(y_train)
    
    
    model = GaussianNB()
    
    model.fit(X_train, y_train)
    
    test = []
    
    for bowler in bowlers_1:
        
        bowl = []
        
        if batsman_data_team[(batsman_data_team['bowler'] == bowler)]['bowler_encoded'].empty:
            continue
        
        encoded_bowler = batsman_data_team[(batsman_data_team['bowler'] == bowler)]['bowler_encoded'].iloc[0]
        
        if batsman_data_team[(batsman_data_team['bowler_encoded'] == encoded_bowler) & (batsman_data_team['balls'] != 0)]['balls'].count() == 0:
            continue
        
        
        avg = batsman_data_team[(batsman_data_team['bowler_encoded'] == encoded_bowler) & (batsman_data_team['balls'] != 0)]['balls'].sum()/batsman_data_team[(batsman_data_team['bowler_encoded'] == encoded_bowler) & (batsman_data_team['balls'] != 0)]['balls'].count()
        bowl.append(avg)
        bowl.append(encoded_bowler)
        
        total_balls = total_balls + avg
        test.append(bowl)
        
        if total_balls > 300 :
            break
        
    if not test:
        #continue
        batsman_data_team['team_encoded'] = batsman_data_team['against'].astype('category').cat.codes
    
        A = batsman_data_team.loc[:, ['balls' , 'team_encoded']].values
        B = batsman_data_team.loc[:, 'runs_scored'].values
        
        X_train, X_test, y_train, y_test = train_test_split(A, B, test_size = 0.01, random_state = 0)
        
        
        Y = np.array(y_train)
        
        
        model = GaussianNB()
        
        model.fit(X_train, y_train)
        
    
    
    predicted= model.predict(test)
    print(batsman)
    print(sum(predicted))
    predicted = list(predicted)
    innings_2_score = innings_2_score + sum(predicted)

print(innings_2_score)
print(total_balls)


from sklearn.metrics import accuracy_score
accuracy_score(y_test,predicted)



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
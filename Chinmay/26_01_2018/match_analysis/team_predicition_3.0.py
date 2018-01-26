# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 19:15:58 2018

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

batsman_all_data = pd.read_csv("all_batsmans_data.csv")


match_type = 'ODI'

team1 = 'India'
team2 = 'Australia'



team_1 = [ 'AM Rahane' , 'RG Sharma' , 'V Kohli' , 'MK Pandey', 'KD Jadhav', 'MS Dhoni' , 'HH Pandya'  , 'B Kumar' , 'JJ Bumrah' , 'Kuldeep Yadav' , 'YZ Chahal']

team_2 = ['H Cartwright' , 'DA Warner' , 'SPD Smith' , 'T Head' , 'GJ Maxwell' , 'M Stoinis' , 'MS Wade' , 'A Agar' , 'P Cummins' , 'NM Coulter-Nile' , 'K Richardson']

bowlers_1 = ['B Kumar' , 'JJ Bumrah' , 'Kuldeep Yadav' , 'HH Pandya' , 'YZ Chahal']

bowlers_2 = ['A Agar' , 'P Cummins' , 'NM Coulter-Nile' , 'KW Richardson', 'M Stoinis', 'T Head']


innings_1_score = 0
total_balls = 0

    
def score_predict(batsman , bowlers , team1 , team2 , match_type):
   
#    batsman = 'SPD Smith'
#    bowlers = bowlers_1
#    team1 = 'Australia'
#    team2 = 'India'
#    match_type = 'ODI'
    
    from sklearn.naive_bayes import GaussianNB
    import numpy as np
    from sklearn.cross_validation import train_test_split
    
    ############ Prediction Based On Bowlers ##############################
    
    batsman_data = batsman_all_data[(batsman_all_data['name'] == batsman)  & (batsman_all_data['match_type'] == match_type)]
    
    batsman_data_team = batsman_data[(batsman_data['against'] == team2)]
    
    if batsman_data_team.empty:
        print("No Data Found against" , team2)
        return 0,0,0,0
    
    batsman_data_team['bowler_encoded'] = batsman_data_team['bowler'].astype('category').cat.codes
    
    A = batsman_data_team.loc[:, ['balls' , 'bowler_encoded']].values
    B = batsman_data_team.loc[:, 'runs_scored'].values
    
    X_train, X_test, y_train, y_test = train_test_split(A, B, test_size = 0.01, random_state = 0)
    
    model = GaussianNB()
    
    model.fit(X_train, y_train)
    
    test = []
    
    for bowler in bowlers:
        
        bowl = []
        #bowlers =  batsman_data_team['bowler'].unique()
        if batsman_data_team[(batsman_data_team['bowler'] == bowler)]['bowler_encoded'].empty:
            continue
        
        encoded_bowler = batsman_data_team[(batsman_data_team['bowler'] == bowler)]['bowler_encoded'].iloc[0]
        
        if batsman_data_team[(batsman_data_team['bowler_encoded'] == encoded_bowler) & (batsman_data_team['balls'] != 0)]['balls'].count() == 0:
            continue
        
        avg = batsman_data_team[(batsman_data_team['bowler_encoded'] == encoded_bowler) & (batsman_data_team['balls'] != 0)]['balls'].sum()/batsman_data_team[(batsman_data_team['bowler_encoded'] == encoded_bowler) & (batsman_data_team['balls'] != 0)]['balls'].count()
        print(batsman_data_team[(batsman_data_team['bowler_encoded'] == encoded_bowler)]['bowler'].iloc[0])
        bowl.append(avg)
        bowl.append(encoded_bowler)
        test.append(bowl)
         
    
    if not test:
        predicted = 0
    else:
        predicted= model.predict(test)
        
    print(batsman , "will score against bowlers" , sum(predicted))
        
    ############### Predicition Based On Against Team Record #########################
    
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
                
            if (df_match.loc[indexs,'batsman'] == batsman):
                runs = runs + df_match.loc[indexs,'runs.batsman']
                balls_faced = balls_faced + 1
                
        bats_data.append(df_match.loc[indexs,'info.match_type'])
        bats_data.append(df_match.loc[indexs,'info.dates'])
        bats_data.append(df_match.loc[indexs,'info.neutral_venue'])
        bats_data.append(home)
        bats_data.append(opposition)
        bats_data.append(runs)
        bats_data.append(balls_faced)
        bats_data.append(df_match.loc[indexs,'info.venue'])
        batsman_data = batsman_data.append(pd.Series(bats_data), ignore_index=True)
        
    batsman_data.columns = ['match_type' , 'date' , 'neutral' ,'home_away', 'against' , 'runs' , 'balls_faced' , 'venue']
    
    

    #batsman_data2 = batsman_data[(batsman_data['match_type'] == match_type) & (batsman_data['home_away'] == home)]
    batsman_data2 = batsman_data[(batsman_data['match_type'] == match_type)]
    batsman_data2['team_encoded'] = batsman_data2['against'].astype('category').cat.codes
    
    A = batsman_data2.loc[:, ['balls_faced' , 'team_encoded']].values
    B = batsman_data2.loc[:, 'runs'].values
    
    X_train, X_test, y_train, y_test = train_test_split(A, B, test_size = 0.01, random_state = 0)

    if batsman_data2[(batsman_data2['against'] == team2)]['team_encoded'].empty:
        print("No Data Found against" , team2)
        return sum(predicted),0,0
        
    
    encoded_team = batsman_data2[(batsman_data2['against'] == team2)]['team_encoded'].iloc[0]
    avg = batsman_data2[(batsman_data2['team_encoded'] == encoded_team) & (batsman_data2['balls_faced'] != 0)]['balls_faced'].sum()/batsman_data2[(batsman_data2['team_encoded'] == encoded_team) & (batsman_data2['balls_faced'] != 0)]['balls_faced'].count()

    model = GaussianNB()
    
    model.fit(X_train, y_train)
    
    test = []
    team = []
    team.append(avg)
    team.append(encoded_team)
    test.append(team)
    predicted2= model.predict(test)
    
    print(batsman , "will score overall" , sum(predicted2))

############### Predicition Based On Against Team Record On Home / Away #########################
    
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
                
            if (df_match.loc[indexs,'batsman'] == batsman):
                runs = runs + df_match.loc[indexs,'runs.batsman']
                balls_faced = balls_faced + 1
                
        bats_data.append(df_match.loc[indexs,'info.match_type'])
        bats_data.append(df_match.loc[indexs,'info.dates'])
        bats_data.append(df_match.loc[indexs,'info.neutral_venue'])
        bats_data.append(home)
        bats_data.append(opposition)
        bats_data.append(runs)
        bats_data.append(balls_faced)
        bats_data.append(df_match.loc[indexs,'info.venue'])
        batsman_data = batsman_data.append(pd.Series(bats_data), ignore_index=True)
        
    batsman_data.columns = ['match_type' , 'date' , 'neutral' ,'home_away', 'against' , 'runs' , 'balls_faced' , 'venue']
    
    

    batsman_data2 = batsman_data[(batsman_data['match_type'] == match_type) & (batsman_data['home_away'] == home)]
    #batsman_data2 = batsman_data[(batsman_data['match_type'] == match_type)]
    batsman_data2['team_encoded'] = batsman_data2['against'].astype('category').cat.codes
    team =  batsman_data2['against'].unique()
    
    A = batsman_data2.loc[:, ['balls_faced' , 'team_encoded']].values
    B = batsman_data2.loc[:, 'runs'].values
    
    X_train, X_test, y_train, y_test = train_test_split(A, B, test_size = 0.01, random_state = 0)

    if batsman_data2[(batsman_data2['against'] == team2)]['team_encoded'].empty:
        print("No Data Found against" , team2)
        return sum(predicted),sum(predicted2),0
        
    
    encoded_team = batsman_data2[(batsman_data2['against'] == team2)]['team_encoded'].iloc[0]
    avg = batsman_data2[(batsman_data2['team_encoded'] == encoded_team) & (batsman_data2['balls_faced'] != 0)]['balls_faced'].sum()/batsman_data2[(batsman_data2['team_encoded'] == encoded_team) & (batsman_data2['balls_faced'] != 0)]['balls_faced'].count()
    
    model = GaussianNB()
    
    model.fit(X_train, y_train)
    
    test = []
    team = []
    team.append(avg)
    team.append(encoded_team)
    test.append(team)
    predicted3= model.predict(test)
    
    print(batsman , "will score based on home/away" , sum(predicted3))
    
    return sum(predicted) , sum(predicted2) , sum(predicted3)


against_bowlers,overall,home_away = score_predict('MS Dhoni' , bowlers_2 , team1 , team2 , match_type)
print(against_bowlers,overall,home_away)



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
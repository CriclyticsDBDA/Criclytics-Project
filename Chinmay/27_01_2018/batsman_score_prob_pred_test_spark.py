# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 23:34:00 2018

@author: Chinmay
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from pyspark.sql.types import StringType
from pyspark import SQLContext , SparkContext
from pyspark.sql.session import SparkSession

sc = SparkContext('local','fisrt_SPARK')  # If using locally
sqlContext = SQLContext(sc)
spark = SparkSession(sc)

# sc.stop()

df = (spark.read.format("csv").options(header="true" , inferSchema = True ).load("final_all_3.0.csv"))

batsman_all_data= (spark.read.format("csv").options(header="true" , inferSchema = True).load("all_batsmans_data_2.0.csv"))

teams = ['Australia' , 'New Zealand' , 'India' , 'Zimbabwe' , 'Bangladesh' , 'South Africa' , 'England' 
         , 'Sri Lanka' , 'Pakistan' , 'West Indies' , 'Ireland']

match_type = 'ODI'

team1 = 'India'
team2 = 'Australia'



team_1 = [ 'AM Rahane' , 'RG Sharma' , 'V Kohli' , 'MK Pandey', 'KD Jadhav', 'MS Dhoni' , 'HH Pandya'  , 'B Kumar' , 'JJ Bumrah' , 'Kuldeep Yadav' , 'YZ Chahal']

team_2 = ['H Cartwright' , 'DA Warner' , 'SPD Smith' , 'T Head' , 'GJ Maxwell' , 'M Stoinis' , 'MS Wade' , 'A Agar' , 'P Cummins' , 'NM Coulter-Nile' , 'K Richardson']


bowlers_1 = ['B Kumar' , 'JJ Bumrah' , 'Kuldeep Yadav' , 'HH Pandya' , 'YZ Chahal']

bowlers_2 = ['A Agar' , 'P Cummins' , 'NM Coulter-Nile' , 'KW Richardson', 'M Stoinis', 'T Head']


innings_1_score = 0
total_balls = 0

from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import train_test_split

from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan,isnull,when,count


def score_predict(batsman , bowlers , team1 , team2 , match_type):
    
    #batsman = 'MS Dhoni'    
    ############ Prediction Based On Bowlers ##############################
    
    batsman_data = batsman_all_data[(batsman_all_data['name'] == batsman)  & (batsman_all_data['match_type'] == match_type)]
    
    batsman_data_team = batsman_data[(batsman_data['against'] == team2)]
    
    batsman_data_team.count()
    
    if batsman_data_team.count() == 0:
        print("No Data Found against" , team2)
        return 0,0,0,0
    
    batsman_data_team = batsman_data_team.toPandas()
    
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
    from pyspark.sql.functions import col
    df_team = df.filter(df['`info.teams`'].rlike('India'))
    #df.filter(col("`info.teams`").isin(['India'])).show()
    
    #match_index = df_team.toPandas()['index_all'].unique()
    match_index = df_team.select('index_all').distinct().rdd.collect()
    match_index
    batsman_data = pd.DataFrame()
    for mindex in match_index:
        
        df_match = df_team.where(col('index_all') == mindex[0])
        
        if df_match[(df_team['batsman']==batsman)].count() == 0:
            continue
        
        runs = 0
        balls_faced = 0
        bats_data = []
        for match_details in df_match.rdd.collect():
            match_details = match_details.asDict()
            ls = pd.Series(match_details['info.teams']).str.strip().apply(ast.literal_eval).apply(pd.Series)
            
            if ls.iloc[0,0] == team1:
                home = 0
                opposition = ls.iloc[0,1]
            else:
                home = 1
                opposition = ls.iloc[0,0]
                
            if (match_details['batsman'] == batsman):
                runs = runs + match_details['runs.batsman']
                balls_faced = balls_faced + 1
                
        bats_data.append(match_details['info.match_type'])
        bats_data.append(match_details['info.dates'])
        bats_data.append(match_details['info.neutral_venue'])
        bats_data.append(home)
        bats_data.append(opposition)
        bats_data.append(runs)
        bats_data.append(balls_faced)
        bats_data.append(match_details['info.venue'])
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
        


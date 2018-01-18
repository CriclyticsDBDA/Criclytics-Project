# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:05:09 2018

@author: dbda
"""

import packages_stat as ps
import numpy as np
import pandas as pd

'''Import dataset'''
df = pd.read_csv("final_all_2.0Scores.csv")
df_t20 = df[(df['info.match_type']=='T20') & (df['info.gender']=='male')]



'''
team_1='India'
team_2='Australia'
team_12='['+"'"+team_1+"'"+', '+"'"+team_2+"'"+']'
team_21='['+"'"+team_2+"'"+', '+"'"+team_1+"'"+']'
df_t20=df_t20[(df_t20['info.teams']==team_12) | (df_t20['info.teams']==team_21)]
'''

#select particular columns from the dataframe
X = df_t20[['info.teams','info.toss.decision','info.toss.winner','info.venue']]
Y = df_t20[['info.outcome.winner']]


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

df_t20.to_csv("df_t20.csv")




Index(['info.dates', 'info.gender', 'info.match_type', 'info.outcome.by.runs',
       'info.outcome.by.wickets', 'info.outcome.result', 'info.outcome.winner',
       'info.player_of_match', 'info.teams', 'info.toss.decision',
       'info.toss.winner', 'info.venue', 'batsman', 'bowler', 'extras.byes',
       'extras.legbyes', 'extras.noballs', 'extras.wides', 'runs.batsman',
       'runs.extras', 'runs.total', 'team', 'wicket.fielders', 'wicket.kind',
       'wicket.player_out'],
      dtype='object')
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 18:25:16 2018

@author: dbda
"""

import pandas as pd
import ast

#df1= pd.read_csv("final_all_1.0Stats.csv")
df1=pd.read_csv("df_all_Final_trimmed.csv")
df1 = df1[((df1['info.match_type']=='T20')|(df1['info.match_type']=='ODI')) & (df1['info.gender']=='male')]


df1['info.dates'] = pd.to_datetime(df1['info.dates'].astype(str).str.findall('\d+').str.join('/'), errors='coerce')
df1=df1.set_index(df1['info.dates'])


df1['info.teams'] = df1['info.teams'].str.strip().apply(ast.literal_eval)
df1['info.player_of_match']=df1['info.player_of_match'].str.strip()
df1['info.player_of_match']=df1['info.player_of_match'].fillna('NA')

df1.to_csv("df_all_Final_trimmed.csv")

#df1['info.outcome.winner']=df1[df1['info.outcome.result']=="no result"]['info.outcome.winner'].fillna('no result')

df1.loc[df1['info.outcome.result']=="no result",'info.outcome.winner']=="no result"






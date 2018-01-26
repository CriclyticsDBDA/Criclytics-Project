# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 00:29:41 2018

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

idx = ndf[ndf.isin(['India' , 'Sri Lanka']).all(1)].index

idx = ndf[(ndf[0]=='India') | (ndf[1]=='India')].index

type(idx)

df = df.loc[idx]

list(ndf)

df_try = df.copy()


match_index = df['index_all'].unique()


batsman_data = pd.DataFrame()
for mindex in match_index:
    df_match = df[(df['index_all']==mindex)].copy()
    
    if df_match[(df['batsman']=='RG Sharma')].empty:
        continue
    runs = 0
    kohli_data = []
    for indexs,match_details in df_match.iterrows():
        batsman = df_match.loc[indexs,'batsman']
        if (df_match.loc[indexs,'batsman'] == 'RG Sharma'):
            runs = runs + df_match.loc[indexs,'runs.batsman']
        #df_match[(df['batsman']=='RG Sharma')]['runs.batsman'].sum()
    print(runs)
    kohli_data.append(df_match.loc[indexs,'info.match_type'])
    kohli_data.append(df_match.loc[indexs,'info.dates'])
    kohli_data.append(df_match.loc[indexs,'info.neutral_venue'])
    kohli_data.append(df_match.loc[indexs,'info.teams'])
    #kohli_data.append(mindex)
    kohli_data.append(runs)
    batsman_data = batsman_data.append(pd.Series(kohli_data ), ignore_index=True)
    print(mindex)
batsman_data.columns = ['match_type' , 'date' , 'neutral' , 'teams' , 'runs']


df_match[(df['batsman']=='RG Sharma')]['runs.batsman'].sum()
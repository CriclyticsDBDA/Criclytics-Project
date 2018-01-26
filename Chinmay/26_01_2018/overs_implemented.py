# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 20:24:07 2018

@author: Chinmay
"""

import pandas as pd
import numpy as np
import ast

#Import dataset
df = pd.read_csv("real_final_all.csv")
#df = df[(df['info.match_type']=='T20')]

teams = ['Australia' , 'New Zealand' , 'India' , 'Zimbabwe' , 'Bangladesh' , 'South Africa' , 'England' 
         , 'Sri Lanka' , 'Pakistan' , 'West Indies' , 'Ireland']


df['info.teams'] = df['info.teams'].str.strip().apply(ast.literal_eval)

ndf = df['info.teams'].apply(pd.Series)


df.date = pd.to_datetime(df.date.astype(str).str.findall('\d+').str.join('/'), errors='coerce')
df['info.dates'] = pd.to_datetime(df['info.dates'].astype(str).str.findall('\d+').str.join('/'), errors='coerce')


#list(df)
df['overs'] = 10
df['over_no'] = 1

df_try = df.copy()

match_index = df['index_all'].unique()

#type(match_index)



for mindex in match_index:
    #print(index)
    df_match = df[(df['index_all']==mindex)].copy()
    #print(df_match)
    #df_match['overs'] = 1
    #df[(df['index_all']==index)] = df_match
    i = 1
    over_no_part = 0
    over_no = 0
    for indexs,match_details in df_match.iterrows():
        #print(index)
        #print(match_details)
        if (i == 1):
            over_no_part = over_no_part + 0.1
            bowler = match_details['bowler']
            team = match_details['team']
            df_match.loc[indexs,'overs'] = over_no_part
            df_match.loc[indexs,'over_no'] = over_no + 1
            #pd.concat([df_try[~df_try.index.isin(df_match.index)], df_match])
            i = i + 1
        else:
            if (team == match_details['team'] ):
                if (bowler == match_details['bowler']):
                    over_no_part = over_no_part + 0.1
                    df_match.loc[indexs,'overs'] = over_no_part
                    df_match.loc[indexs,'over_no'] = over_no + 1
                    #pd.concat([df_try[~df_try.index.isin(df_match.index)], df_match])
                    i = i + 1
                else:
                    bowler = match_details['bowler']
                    over_no =over_no + 1
                    over_no_part = over_no
                    over_no_part = over_no_part + 0.1
                    df_match.loc[indexs,'overs'] = over_no_part
                    df_match.loc[indexs,'over_no'] = over_no + 1
                    #pd.concat([df_try[~df_try.index.isin(df_match.index)], df_match])
                    i = i + 1
            else:
                over_no_part = 0
                over_no = 0
                over_no_part = over_no_part + 0.1
                bowler = match_details['bowler']
                team = match_details['team']
                df_match.loc[indexs,'overs'] = over_no_part
                df_match.loc[indexs,'over_no'] = over_no + 1
                #pd.concat([df_try[~df_try.index.isin(df_match.index)], df_match])
                i = i + 1
    #df[(df['index_all']==index)].update(df_match)
    #df.update(df_match)
        #pd.concat([df[~df.index.isin(df_match.index)], df_match])
    print(over_no + 1)
    print(mindex)
    #break
    df_try[(df_try['index_all']==mindex)] = df_match
    

df_try.to_csv('final_all_3.0.csv')

#pd.concat([df_try[~df_try.index.isin(df_match.index)], df_match])

#df_match.index

#df_try['index_all'].map(df_match.set_index('index_all')['overs'])


#df[(df['index_all']==mindex)] = df_match
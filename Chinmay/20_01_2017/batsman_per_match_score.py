# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 23:01:35 2018

@author: Chinmay
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 20:24:07 2018

@author: Chinmay
"""

import pandas as pd
import numpy as np
import ast

#Import dataset
df = pd.read_csv("final_all_2.0.csv")
df = df[(df['info.match_type']=='T20')]

teams = ['Australia' , 'New Zealand' , 'India' , 'Zimbabwe' , 'Bangladesh' , 'South Africa' , 'England' 
         , 'Sri Lanka' , 'Pakistan' , 'West Indies' , 'Ireland']


df['info.teams'] = df['info.teams'].str.strip().apply(ast.literal_eval)

ndf = df['info.teams'].apply(pd.Series)


idx = ndf[ndf.isin(['India' , 'Sri Lanka']).all(1)].index

df = df.loc[idx]

list(df)
df['overs'] = 10
df['over_no'] = 1

df_try = df.copy()

batsman_data = pd.DataFrame(columns = ['match_type' , 'match_no' , 'runs'])

kohli_data = []

match_index = df['index_all'].unique()

type(match_index)

new_match_index = match_index

new_match_index = np.delete(new_match_index,0)

batsman_data = pd.DataFrame()
for mindex in new_match_index:
    df_match = df[(df['index_all']==mindex)].copy()
    
    if df_match[(df['batsman']=='V Kohli')].empty:
        continue
    runs = 0
    kohli_data = []
    for indexs,match_details in df_match.iterrows():
        batsman = df_match.loc[indexs,'batsman']
        if (df_match.loc[indexs,'batsman'] == 'V Kohli'):
            runs = runs + df_match.loc[indexs,'runs.batsman']
    print(runs)
    kohli_data.append(df_match.loc[indexs,'info.match_type'])
    kohli_data.append(df_match.loc[indexs,'info.dates'])
    kohli_data.append(df_match.loc[indexs,'info.neutral_venue'])
    kohli_data.append(df_match.loc[indexs,'info.teams'])
    #kohli_data.append(mindex)
    kohli_data.append(runs)
    batsman_data = batsman_data.append(pd.Series(kohli_data ), ignore_index=True)
    break
    print(mindex)
    

df_try.to_csv('final_all_2.0.csv')

pd.concat([df_try[~df_try.index.isin(df_match.index)], df_match])

df_match.index

df_try['index_all'].map(df_match.set_index('index_all')['overs'])


df[(df['index_all']==mindex)] = df_match
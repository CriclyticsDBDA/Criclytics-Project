# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:44:46 2018

@author: dbda
"""
import pandas as pd
import numpy as np
import glob
import yaml

df_all_stats=pd.read_csv('final_all_1.0Stats.csv',low_memory=False)

all_match_type_list=[]

for i in df_all_stats['info.match_type']:
    all_match_type_list.append(i)

all_files = glob.glob('D:\\Dbda\\Project\\Raw_Data\\all\\*.yaml')
file_count = 1
all_score_list=[]
type_counter=0


for file_name in all_files:
    try:
        file_count += 1
        d = yaml.load(open(file_name))
        for i in d['innings']:
            df_scores = pd.DataFrame(i[list(i.keys())[0]])
            df_scores = pd.io.json.json_normalize(pd.DataFrame(df_scores.deliveries.tolist()).stack().tolist()).assign(team=df_scores.team,type=all_match_type_list[type_counter]
)
            all_score_list.append(df_scores)
            print(file_count)

        if type_counter < 3994:
            type_counter += 1
        else:
            break
    except:
        print(file_name)

df_all_scores=pd.concat(all_score_list)
df_all_scores.to_csv('final_all_3.0Scores.csv')

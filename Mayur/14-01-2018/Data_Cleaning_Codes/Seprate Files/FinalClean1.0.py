import yaml
import pandas as pd
import numpy as np
import glob

all_stats_list=[]
all_score_list=[]
file_count=0

'''Get All the file_names into List'''
all_files=glob.glob('D:\\Dbda\\Project\\Data\\all\\*.yaml')

'''Get Only Statistics Data into Stat DataFrame
for file_name in all_files:
    with open(file_name, 'r') as f:
        file_count += 1
        print(file_count)
        df_stats = pd.io.json.json_normalize(yaml.load(f))
        print(df_stats)
        all_stats_list.append(df_stats)


df_all_stats=pd.concat(all_stats_list)
df_all_stats=df_all_stats[df_all_stats.columns.difference(['innings'])]
df_all_stats.drop(columns=['meta.created','meta.data_version','meta.revision'],inplace=True)


df_all_stats.columns=['city', 'date', 'gender', 'match_type',
       'outcome.by.runs', 'outcome.winner', 'overs',
       'player_of_match', 'teams', 'decision',
       'winner', 'umpires', 'venue']




df_all_stats.to_csv('final_all_1.0Stats.csv')
'''


'''Get only Innings_Score Data into Innings_Score Dataframe'''


file_count=0
for file_name in all_files:
    try:
        file_count += 1
        print(file_count)
        d = yaml.load(open(file_name))
       # print(file_name)
        for i in d['innings']:
            df_scores = pd.DataFrame(i[list(i.keys())[0]])
            df_scores = pd.io.json.json_normalize(pd.DataFrame(df_scores.deliveries.tolist()).stack().tolist()).assign(team=df_scores.team)
            all_score_list.append(df_scores)
    except:
        print(file_name)

df_all_scores=pd.concat(all_score_list)
df_all_scores.to_csv('final_all_1.0Scores.csv')


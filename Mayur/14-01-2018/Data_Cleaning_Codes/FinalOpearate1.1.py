import pandas as pd
import numpy as np
import glob
import yaml

df_all_stats=pd.read_csv('final_all_1.0Stats.csv',low_memory=False)
df_all_scores=pd.read_csv('final_all_2.0Scores.csv',low_memory=False)

'''
df_all_stats.columns=['city', 'date', 'gender', 'match_type',
       'outcome.by.runs', 'outcome.winner', 'overs',
       'player_of_match', 'teams', 'decision',
       'winner', 'umpires', 'venue']
'''
all_dates_list=[]
all_venue_list=[]
for i in df_all_stats['info.dates']:
    all_dates_list.append(i)

for i in df_all_stats['info.venue']:
    all_venue_list.append(i)

all_files = glob.glob('D:\\Dbda\\Project\\Raw_Data\\all\\*.yaml')
file_count = 1
all_score_list=[]
date_counter=0


for file_name in all_files:
    try:
        file_count += 1
        d = yaml.load(open(file_name))
        for i in d['innings']:
            df_scores = pd.DataFrame(i[list(i.keys())[0]])
            df_scores = pd.io.json.json_normalize(pd.DataFrame(df_scores.deliveries.tolist()).stack().tolist()).assign(team=df_scores.team,date=all_dates_list[date_counter],venue=all_venue_list[date_counter]
)
            all_score_list.append(df_scores)
            print(file_count)

        if date_counter < 3994:
            date_counter += 1
        else:
            break
    except:
        print(file_name)

df_all_scores=pd.concat(all_score_list)
df_all_scores.to_csv('final_all_ENDScores.csv')

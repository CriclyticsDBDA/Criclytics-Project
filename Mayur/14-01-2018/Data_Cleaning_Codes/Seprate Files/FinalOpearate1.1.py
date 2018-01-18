import pandas as pd
import numpy as np
import glob
import yaml

df_all_stats=pd.read_csv('final_all_1.0Stats.csv',low_memory=False)
"""df_all_scores=pd.read_csv('D://Dbda//Project//Clean_Data//final_all_2.0Scores.csv',low_memory=False)"""

all_match_type_list=[]
for i in df_all_stats['info.match_type']:
    all_match_type_list.append(i)

all_dates_list=[]
for i in df_all_stats['info.dates']:
    all_dates_list.append(i)


all_outcome_runs=[]
for i in df_all_stats['info.outcome.by.runs']:
    all_outcome_runs.append(i)

all_outcome_wickets=[]
for i in df_all_stats['info.outcome.by.wickets']:
    all_outcome_wickets.append(i)
    
all_outcome_winner=[]
for i in df_all_stats['info.outcome.winner']:
    all_outcome_winner.append(i)

all_info_teams=[]
for i in df_all_stats['info.teams']:
    all_info_teams.append(i)

all_info_toss_decisions=[]
for i in df_all_stats['info.toss.decision']:
    all_info_toss_decisions.append(i)

all_info_toss_winner=[]
for i in df_all_stats['info.toss.winner']:
    all_info_toss_winner.append(i)

all_info_venue=['']
for i in df_all_stats['info.venue']:
    all_info_venue.append(i)


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
            df_scores = pd.io.json.json_normalize(pd.DataFrame(df_scores.deliveries.tolist()).stack().tolist()).assign(team=df_scores.team,date=all_dates_list[type_counter],type=all_match_type_list[type_counter],outcome_runs_win=all_outcome_runs[type_counter],
                                                 outcome_wickets_win=all_outcome_wickets[type_counter],outcome_winner=all_outcome_winner[type_counter],info_teams=all_info_teams[type_counter],info_toss_decisions=all_info_toss_decisions[type_counter],
                                                 info_toss_winner=all_info_toss_winner[type_counter],info_venue=all_info_venue[type_counter])
            all_score_list.append(df_scores)
            print(file_count)

        if type_counter < 3994:
            type_counter += 1
        else:
            break
    except:
        print(file_name)

df_all_scores=pd.concat(all_score_list)
df_all_scores.to_csv('final_all_5.0Scores.csv')

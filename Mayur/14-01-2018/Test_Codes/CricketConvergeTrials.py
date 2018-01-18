import yaml
import pandas as pd
import numpy as np
import glob


#Reading all the files from the folder
all_yaml_files=glob.glob('D:\\Dbda\\Project\\Test_Data\\*.yaml')
list_all_yaml_files=[]
file_count=0

'''
for yaml_file_name in all_yaml_files:               #Getting all .yaml file names
    file_count+=1
    print(yaml_file_name)
    
print(file_count)
'''
all_stat_list=[]

#Getting all .yaml file names
for yaml_file_name in all_yaml_files:
    with open(yaml_file_name, 'r') as f:
        df_all_stats = pd.io.json.json_normalize(yaml.load(f))
        df2_all_stats = df_all_stats[df_all_stats.columns.difference(['innings'])]
        all_stat_list.append(df2_all_stats)
        print(all_stat_list)


'''
#renaming columns
df2_all_stats.columns=['match.city', 'match.dates', 'match.gender', 'match.match_type',
       'match.outcome.by.runs', 'match.outcome.winner', 'match.overs',
       'match.player_of_match', 'match.teams', 'match.toss.decision',
       'match.toss.winner', 'match.umpires', 'match.venue']
'''

df2_all_stats = pd.concat(all_stat_list)
df2_all_stats.to_csv('test_matchstat_all_matches.csv')



#Seprate Innings Score in different lists and concat them into DF
df_list = []




for yaml_file_name in all_yaml_files:
    d = yaml.load(open(yaml_file_name))
    for i in d['innings']:
        df = pd.DataFrame(i[list(i.keys())[0]])
        df = pd.io.json.json_normalize(pd.DataFrame(df.deliveries.tolist()).stack().tolist()).assign(team=df.team)

    #    df.at['Total','runs.total']=df['runs.total'].sum()              #calculate total innings wise runs
     #   df.at['Total', 'wicket.kind'] = df['wicket.kind'].count()        #calculate total wickets innings wise
        df_list.append(df)

final_score = pd.concat(df_list)

#save result to local file
final_score.to_csv('Test_cricket_all_scores.csv')






















'''
#Get Batsman details
batsman_name=input("Enter Batsmen Name:- ")
condition_batsman=final_score.loc[final_score['batsman']==batsman_name]
player_total_runs=condition_batsman['runs.batsman'].sum()
print(player_total_runs)

#Get Bowler Details
bowler_name=input("Enter Bowler Name:- ")
condition_bowler=final_score.loc[final_score['bowler']==bowler_name]
bowler_total_runs=condition_bowler['runs.total'].sum()
bowler_total_extras=condition_bowler['runs.extras'].sum()
bowler_total_overs=round(condition_bowler['team'].count()/6,0)
bowler_total_wickets=condition_bowler['wicket.kind'].count()
print("Bowler: {0}, Overs : {1}, Runs: {2}, Extras: {3}, Wickets: {4}".format(bowler_name,bowler_total_overs,bowler_total_runs,bowler_total_extras,bowler_total_wickets))

#save result to local file
final_score.to_csv('cricket.csv')
final_stat.to_csv('matchstat.csv')
'''






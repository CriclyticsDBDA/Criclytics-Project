import yaml
import pandas as pd
import numpy as np

with open('C:\\Users\\dbda\\Downloads\\211028.yaml', 'r') as f:
    df = pd.io.json.json_normalize(yaml.load(f))

#exclude columns of innings score
df2 = df[df.columns.difference(['innings'])]

#Seprate Innings Score in different lists and concat them into DF
d = yaml.load(open('D:\\Dbda\\Project\\t20s\\211028.yaml'))
df_list = []


#get Metadata stats into different dataframe
match_stat=[]
match_stat.append(df2)

for i in d['innings']:
    df = pd.DataFrame(i[list(i.keys())[0]])
    df = pd.io.json.json_normalize(pd.DataFrame(df.deliveries.tolist()).stack().tolist()).assign(team=df.team)

    df.at['Total','runs.total']=df['runs.total'].sum()              #calculate total innings wise runs
    df.at['Total', 'wicket.kind'] = df['wicket.kind'].count()        #calculate total wickets innings wise
    df_list.append(df)

final_score = pd.concat(df_list)
final_stat =pd.concat(match_stat)

#renaming columns
final_stat.columns=['match.city', 'match.dates', 'match.gender', 'match.match_type',
       'match.outcome.by.runs', 'match.outcome.winner', 'match.overs',
       'match.player_of_match', 'match.teams', 'match.toss.decision',
       'match.toss.winner', 'match.umpires', 'match.venue']

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







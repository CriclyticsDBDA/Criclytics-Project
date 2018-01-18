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

for i in d['innings']:
    df = pd.DataFrame(i[list(i.keys())[0]])
    df = pd.io.json.json_normalize(
         pd.DataFrame(df.deliveries.tolist()).stack().tolist()
    ).assign(team=df.team)
    # df.loc['Total'] = pd.Series(df['runs.total'].sum(), index=['runs.total'])
    # df.loc['Total'] = pd.Series(df['wicket.kind'].count(), index=['wicket.kind'])

    df_list.append(df)


final_df = pd.concat(df_list)
final_df['match_city']=df2['info.city']
final_df['match_date']=df2['info.dates']
final_df['match_type']=df2['info.match_type']
final_df['match_winner']=df2['info.outcome.winner']
final_df['match_overs']=df2['info.overs']
final_df['match_player']=df2['info.player_of_match']
final_df['match_teams']=df2['info.teams']
final_df['match_toss']=df2['info.toss.winner']
final_df['match_stadium']=df2['info.venue']

final_df.loc['Total'] = pd.Series(final_df['runs.total'].sum(), index = ['runs.total'])


batsmen= "A Flintoff"
bowler = "D Gough"



final_df.to_csv('cricekt.csv')





# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:17:45 2018

@author: Author
"""
import pandas as pd

class packages_stats:
    def __init__():
        pass
    
    def total_runs(dataframe,name):
        df_Batsman=dataframe[dataframe['batsman']==name]
        runs_Total=df_Batsman['type'].sum()
        runs_Total
        
    def average_runs(dataframe,name):
        df_Batsman=dataframe[dataframe['batsman']==name]
        numberOfOuts=df_Batsman['wicket.kind'].count()
        runs_Total=df_Batsman['type'].sum()
        batting_Average= runs_Total/numberOfOuts
        batting_Average
        
    def average_strikeRate(dataframe,name):
        df_Batsman=dataframe[dataframe['batsman']==name]
        runs_Total=df_Batsman['type'].sum()
        ball_Faced=df_Batsman['runs.total'].count()
        strikeRate=(runs_Total*100)/ball_Faced
        strikeRate
        
    def average_wickets(dataframe,name):
        df_bowler=dataframe[dataframe['bowler']==name]
        legal_Wickets=df_bowler[(df_bowler['wicket.kind']=="bowled") | (df_bowler['wicket.kind']=="caught") |
                                (df_bowler['wicket.kind']=="caught and bowled") | (df_bowler['wicket.kind']=="hit wicket") |
                                (df_bowler['wicket.kind']=="lbw") | (df_bowler['wicket.kind']=="stumped") ]
        total_dates=legal_Wickets.date.unique()
        dates_count= pd.DataFrame(total_dates).count()
        total_wickets=legal_Wickets['wicket.kind'].count()
        average_wickets=total_wickets/dates_count[0]
        average_wickets
        
    def total_wickets(dataframe,name):
        df_bowler=dataframe[dataframe['bowler']==name]
        legal_Wickets=df_bowler[(df_bowler['wicket.kind']=="bowled") | (df_bowler['wicket.kind']=="caught") |
                                (df_bowler['wicket.kind']=="caught and bowled") | (df_bowler['wicket.kind']=="hit wicket") |
                                (df_bowler['wicket.kind']=="lbw") | (df_bowler['wicket.kind']=="stumped") ]
        total_wickets=legal_Wickets['wicket.kind'].count()
        total_wickets
        
    def economy_rate(dataframe,name):
        df_bowler=dataframe[dataframe['bowler']==name]
        df_noBalls=df_bowler['extras.noballs'].count()
        df_wideBalls=df_bowler['extras.wides'].count()
        df_totalBalls=df_bowler['runs.total'].count() 
        df_totalRuns=df_bowler['runs.total'].sum()
        df_actual_Balls=df_totalBalls-df_wideBalls-df_noBalls
        eco_rate=df_totalRuns/(df_actual_Balls/6)
        eco_rate
        
    def visualize_tree(tree, feature_names):
    #Create tree png using graphviz
        import subprocess
        from sklearn.tree import export_graphviz
        with open("dt.dot", 'w') as f:
            export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

        command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
        try:
            subprocess.check_call(command)
        except:
            exit("Could not run dot, ie graphviz, to "
             "produce visualization")
      
        
        

        
        
        
        
        
        
        

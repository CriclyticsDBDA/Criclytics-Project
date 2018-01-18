"""
Created on Mon Dec 18 17:00:54 2017

@author: Mayur
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#df_stats = pd.read_csv("finalt201.0Stats.csv",index_col='info.dates')

df_scores= pd.read_csv("finalt201.0Scores.csv")
df_scores.date = pd.to_datetime(df_scores.date.astype(str).str.findall('\d+').str.join('/'), errors='coerce')
df_scores.set_index(df_scores['date'])

df_batsman=df_scores[df_scores['batsman']=="V Kohli"]

score=df_batsman.groupby('date').sum()
score_year=score.resample('A').sum()

score_year.index
score_year['runs.total']
plt.plot(score_year.index,score_year['runs.total'])


sns.barplot(score_year.index,score_year['runs.total'])












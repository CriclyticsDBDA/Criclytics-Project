# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 20:49:04 2018

@author: dbda
"""
import packages_stat as ps
import numpy as np
import pandas as pd

df = pd.read_csv("final_all_4.0Scores.csv")
df_t20 = df[(df['type']=='T20') & (df['team']=='India')]

from sklearn import tree




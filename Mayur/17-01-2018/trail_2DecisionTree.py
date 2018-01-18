import pandas as pd
import numpy as np
from packages_stat import packages_stats

#Import dataset
df = pd.read_csv("final_all_2.0Stats.csv")
df = df[(df['info.match_type']=='T20') & (df['info.gender']=='male')]

#choosing particular teams
team_1='India'
team_2='Australia'
team_12='['+"'"+team_1+"'"+', '+"'"+team_2+"'"+']'
team_21='['+"'"+team_2+"'"+', '+"'"+team_1+"'"+']'
df=df[(df['info.teams']==team_12) | (df['info.teams']==team_21)]

#getting only the columns that are required
df=df[['info.venue','info.toss.decision','info.toss.winner','info.outcome.winner']]

#Using LabelEncoder and OneHotEncoder to encode data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
onehotencoder=OneHotEncoder()
df['info.toss.decision'] = labelencoder.fit_transform(df['info.toss.decision'])
df['info.toss.winner']= labelencoder.fit_transform(df['info.toss.winner'])
df['info.outcome.winner']= labelencoder.fit_transform(df['info.outcome.winner'])

#OneHotencoding of Venue 
df = pd.concat([df, pd.get_dummies(df['info.venue'])], axis=1)
df=df.drop('info.venue',axis=1)


#select particular columns from the dataframe
X = df.drop(['info.outcome.winner'],axis=1)
Y = df[['info.outcome.winner']]
feature_name=X.columns


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)

'''criterion : string, optional (default=”gini”)
 The function to measure the quality of a split. 
 Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
'''

#predicting with test dataset
y_pred = classifier.predict(X_test)

 
#Calling visualize tree from package stat.
packages_stats.visualize_tree(classifier,feature_name)

'''
#plotting GraphTree to measure performance of Tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()

export_graphviz(classifier, out_file=dot_data, feature_names= feature_name,
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
'''    
        

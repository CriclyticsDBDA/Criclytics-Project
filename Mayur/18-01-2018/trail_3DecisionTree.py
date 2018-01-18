import pandas as pd
import numpy as np
import ast
from packages_stat import packages_stats

#Import dataset
df = pd.read_csv("final_all_2.0Stats.csv")
df = df[(df['info.match_type']=='T20') & (df['info.gender']=='male')]

teams=['Australia', 'New Zealand',
       'India', 'Zimbabwe', 'Bangladesh', 'South Africa', 'England',
       'Sri Lanka', 'Pakistan', 'West Indies',
       'Ireland']


#select the teams that are required
df['info.teams'] = df['info.teams'].str.strip().apply(ast.literal_eval)
ndf = df['info.teams'].apply(pd.Series)
idx = ndf[ndf.isin(['India','Australia']).all(1)].index

df=df.loc[idx]

#getting only the columns that are required
df=df[['info.venue','info.toss.decision','info.toss.winner','info.outcome.winner']]

#specifying where one and zero will be there.
'''Even LabelEncoder and OnehotEncoder can be used but for better User readability 
    map function can be used.'''
'''Indian team winning is given by 1 and losing is given by 0.'''
 
df['info.toss.decision'] = df['info.toss.decision'].map({'bat': 1, 'field': 0})
df['info.toss.winner']=df['info.toss.winner'].map({teams[2]:1,teams[0]:0})
df['info.outcome.winner']=df['info.outcome.winner'].map({teams[2]:1,teams[0]:0})


#OneHotencoding of Venue 
'''since number of Venues are high in numbers we have to use one hot encoder'''
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

#checking accuracy of model
'''Accuracy Score'''
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)


'''Confusion Matrix
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp=confusion_matrix(y_test,y_pred).ravel()
"True_Positive= {0} , False_Positive= {1},False_Negative= {2},True_Positive={3}".format(tn,fp,fn,tp)
'''

#Calling visualize tree from package stat.
packages_stats.visualize_tree(classifier,feature_name)

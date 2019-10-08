#!pip3 install gunicorn 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Read the data into a pandas Dataframe
df = pd.read_csv('/home/cdsw/170_flask_app/ml-flask-tutorial/sample_data.csv')

df.head(10)


# In our sample data, we have data about airline delays, with the following columns: 
# * ORIGIN (Origin Airport)
# * DEST (Destination Airport)
# * UNIQUE_CARRIER (Airline  Carrier)
# * DAY_OF_WEEK (Day of the Week)
# * DEP_HOUR (Hour of Departure)
# * ARR_DELAY (Arrival Delay in minutes)
# 
# We will build a model to predict whether a flight is delayed more than 5 minutes or not, given the ORIGIN, DEST and UNIQUE_CARRIER


# First, we transform ARR_DELAY into a 1/0 format for Delay/No Delay
# For this we are going to use the Python Lambda function on the dataframe

df['ARR_DELAY'] = df['ARR_DELAY'].apply(lambda x:1 if x>=5 else 0)

sns.countplot(x='ARR_DELAY', data=df,palette='RdBu_r')

#Convert Categorical Variables into Dummy Variables
df = pd.concat([df,pd.get_dummies(df['UNIQUE_CARRIER'],drop_first=True,prefix="UNIQUE_CARRIER")],axis=1)
df = pd.concat([df,pd.get_dummies(df['ORIGIN'],drop_first=True,prefix="ORIGIN")],axis=1)
df = pd.concat([df,pd.get_dummies(df['DEST'],drop_first=True,prefix="DEST")],axis=1)
df = pd.concat([df,pd.get_dummies(df['DAY_OF_WEEK'],drop_first=True,prefix="DAY_OF_WEEK")],axis=1)
df = pd.concat([df,pd.get_dummies(df['DEP_HOUR'],drop_first=True,prefix="DEP_HOUR")],axis=1)

#Drop the original Categorical Variables
df.drop(['ORIGIN','DEST','UNIQUE_CARRIER','DAY_OF_WEEK','DEP_HOUR'],axis=1,inplace=True)

#Create the train and test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('ARR_DELAY',axis=1), 
                                                    df['ARR_DELAY'], test_size=0.30, 
                                                    random_state=101)

from sklearn.linear_model import LogisticRegression

#Train the model
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

#Predicting on the Test Set
predictions = logmodel.predict(X_test)

#Model Evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

truePos = X_test[((predictions == 1) & (y_test == predictions))]
falsePos = X_test[((predictions == 1) & (y_test != predictions))]
trueNeg = X_test[((predictions == 0) & (y_test == predictions))]
falseNeg = X_test[((predictions == 0) & (y_test != predictions))]

TP = truePos.shape[0]
FP = falsePos.shape[0]
TN = trueNeg.shape[0]
FN = falseNeg.shape[0]

accuracy = float(TP + TN)/float(TP + TN + FP + FN)
print('Accuracy: '+str(accuracy))


# The model has an overall accuracy of 0.61, which is not too bad given the limited dataset on which we trained the model. We will not try to improve on the model here, as that is not the objective of this tutorial!

# ## Saving the Model using Pickle

import pickle

with open('/home/cdsw/models/logmodel.pkl', 'wb') as fid:
    pickle.dump(logmodel, fid,2)  

#Save a dictionary of the index keys to make the dummy variables out of user input

#create a dataframe containing only the categorical variables. In our case, it is the entire dataset except the ARR_DELAY column
cat = df.drop('ARR_DELAY',axis=1)
index_dict = dict(zip(cat.columns,range(cat.shape[1])))

#Save the index_dict into disk
with open('cat', 'wb') as fid:
    pickle.dump(index_dict, fid,2)  



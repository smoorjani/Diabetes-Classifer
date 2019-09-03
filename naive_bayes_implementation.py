import numpy as np
import pandas as pd

# Naive Bayes Implementation

df = pd.read_csv('diabetes.csv')

# Scaling data to ensure equal consideration
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('Outcome',axis=1))
scaled = scaler.transform(df.drop('Outcome',axis=1))

df_columns = df.columns[:1]
df_columns = df_columns.append(df.columns[2:])

scaled_df = pd.DataFrame(scaled,columns=df_columns)

# Creating our training and testing sets
 
from sklearn.cross_validation import train_test_split

X = df.drop(labels=['Outcome'],axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Training and testing the model

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print('\n')
print(classification_report(y_test,y_pred))  






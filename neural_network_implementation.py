import numpy as np
import pandas as pd

# Neural Network Implementation

df = pd.read_csv('diabetes.csv')

# Scaling our dataset for equal consideration of all data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('Outcome',axis=1))
scaled = scaler.transform(df.drop('Outcome',axis=1))

df_columns = df.columns[:1]
df_columns = df_columns.append(df.columns[2:])

scaled_df = pd.DataFrame(scaled,columns=df_columns)

# Creating our training/testing sets
from sklearn.cross_validation import train_test_split
X = df.drop(labels=['Outcome'],axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Training our neural network (more detail found on presentation)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 8))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 20, nb_epoch = 500)
y_pred = classifier.predict(X_test)

'''
print(confusion_matrix(y_test,y_pred))
print('\n')
print(classification_report(y_test,y_pred))  
'''





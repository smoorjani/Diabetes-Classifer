import numpy as np
import pandas as pd

df = pd.read_csv('diabetes.csv')
print(df)

# Scaling data for equal consideration

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('Outcome',axis=1))
scaled = scaler.transform(df.drop('Outcome',axis=1))

print(scaled)
df_columns = df.columns[:1]
df_columns = df_columns.append(df.columns[2:])

scaled_df = pd.DataFrame(scaled,columns=df_columns)
 print(scaled_df)

# Creating training and testing sets

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

X = df.drop(labels=['Outcome'],axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
test_error = []


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report

# Testing various k values to determine optimal k

n = 1
m = 50
for k in range(n,m):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train.astype('int'))
    prediction = knn.predict(X_test)
    test_error.append(1 - np.mean(prediction != y_test))
    '''
    print(confusion_matrix(y_test.astype('int'),prediction))
    print('\n')
    print(classification_report(y_test.astype('int'),prediction))
    '''

# Plotting the k value against accuracy

import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(range(n,m),test_error,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Accuracy vs K Value')
plt.xlabel('K')
plt.ylabel('Accuracy Rate')
plt.savefig('accuracy.png')

# Fitting a k to our model and training it

knn = KNeighborsClassifier(n_neighbors=40)
knn.fit(X_train,y_train.astype('int'))

pred = knn.predict(X_test)

print(test_error)




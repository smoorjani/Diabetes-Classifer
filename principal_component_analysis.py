import pandas as pd
import numpy as np

df = pd.read_csv('diabetes.csv',usecols=[i for i in range(8)])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled = scaler.transform(df)

df_columns = df.columns[:8]

scaled_df = pd.DataFrame(scaled,columns=df_columns)

from sklearn.decomposition import PCA
from sklearn import preprocessing

scaled_df = preprocessing.scale(scaled_df.T)

pca = PCA()
pca.fit(scaled_df)
pca_df = pca.transform(scaled_df)

percentage_variation = np.round(pca.explained_variance_ratio_*100,decimals = 2)

labels = df.columns

import seaborn as sns
import matplotlib.pyplot as plt


ax = sns.barplot(x=list(range(1,len(percentage_variation)+1)),y=percentage_variation)
ax.set_xticklabels(labels,rotation=90)

#plt.bar(x=range(1,len(percentage_variation)+1),height=percentage_variation,tick_label=labels)
plt.xlabel('Principal Components')
plt.ylabel('Percentage of Explained Variance')
plt.title('Scree Plot of PCA on our Dataset')
plt.tight_layout()
plt.savefig('pca.png')
print(labels)

PCA_labels = ['PC'+str(x) for x in range(1,len(percentage_variation)+1)]
pca_dfa = pd.DataFrame(pca_df,index=labels,columns=PCA_labels)

plt.scatter(pca_dfa.PC1,pca_dfa.PC2)
plt.title('Correlation in Data based on PCA')
plt.xlabel('PC1 - {0}%'.format(percentage_variation[0]))
plt.ylabel('PC2 - {0}%'.format(percentage_variation[1]))

for val in pca_dfa.index:
    plt.annotate(val,(pca_dfa.PC1.loc[val],pca_dfa.PC2.loc[val]))
    
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#df = pd.read_csv("/Users/apple/Downloads/K-Means-Cluster/kaggle_Interests_group.csv")
df = pd.read_csv("/Users/apple/Downloads/K-Means-Cluster/child_growth_pci.csv")
df.columns
df.isna().sum()
df.fillna(0, inplace= True)
x = df.iloc[:,5:]
print(x)

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
  model = KMeans(n_clusters = i, init = "k-means++")
  model.fit(x)
  wcss.append(model.inertia_)


plt.figure(figsize=(10,10))
plt.plot(range(1,11), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

from sklearn.decomposition import PCA
pca = PCA(2)

data = pca.fit_transform(x)

plt.figure(figsize=(10,10))
var = np.round(pca.explained_variance_ratio_*100, decimals = 1)
lbls = ['PC'+ str(x) for x in range(1,len(var)+1)]
plt.bar(x=range(1,len(var)+1), height = var, tick_label = lbls)
plt.ylabel('Variance')
plt.show()

model1 = KMeans(n_clusters = 4, init = "k-means++")
label = model1.fit_predict(data)
print(label)

model2 = KMeans(n_clusters = 6, init = "k-means++")
y2 = model2.fit_predict(x)

plt.figure(figsize=(15,15))
uniq = np.unique(label)
for i in uniq:
  plt.scatter(data[label == i , 0] , data[label == i , 1] , label = i)
  
plt.legend()
plt.show()

label2 = model2.fit_predict(data)

plt.figure(figsize=(15,15))
uniq = np.unique(label2)
for i in uniq:
  plt.scatter(data[label2 == i , 0] , data[label2 == i , 1] , label = i)
plt.xlabel([])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

centers = np.array(model2.cluster_centers_)
plt.figure(figsize=(15,15))
uniq = np.unique(label2)

for i in uniq:
  plt.scatter(data[label2 == i , 0] , data[label2 == i , 1] , label = i)
plt.xlabel([])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(centers[:,0], centers[:,1], marker="x", color='k')
plt.legend()
plt.show()

cleandf= df.iloc[:,5:]
correlation = cleandf.corr()
plt.figure(figsize=(30,15))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')
plt.title('Correlation between different fearures')
plt.show()

sns.pairplot(cleandf, diag_kind="kde")
plt.title('Pair plot Analysis between different fearures')
plt.show()


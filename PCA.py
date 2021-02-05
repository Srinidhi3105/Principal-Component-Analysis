import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

data = pd.read_csv("wine.csv")
data.columns
data.head()

#standardizing the values
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(data)

scaled_data = scaler.transform(data)
scaled_data


from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)

scaled_data.shape
x_pca.shape

#plotting first 2 principle components
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=data['Type'])
plt.xlabel('First principle component')
plt.ylabel('Second principle component')


#Performing Clustering  on full data

#Kmeans on full data

#plotting the graph to find the best number of clusters , using the elbow method
K = range(1,10)
Sum_of_squared_distances = []
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(scaled_data)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K,Sum_of_squared_distances,'bx-')
plt.xlabel('No of Clusters')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method for Optimal K')
plt.show()

km = KMeans(n_clusters=4)
km = km.fit(scaled_data)


#Performing hierarchiel clustering on full data
dendogram = sch.dendrogram(sch.linkage(data,method='ward'))

hc = AgglomerativeClustering(n_clusters=16,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(data)


#peforming clustering on Princple component scores

#Kmeans on PCA scores
K = range(1,10)
Sum_of_squared_distances = []

for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(x_pca)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K,Sum_of_squared_distances,'bx-')
plt.xlabel('No of Clusters')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method for Optimal K')
plt.show()

#number of clusters for the principal components is 3
km = KMeans(n_clusters=4)
km = km.fit(x_pca)

#Performing hierachial clustering on PCA scores
dendogram_pca = sch.dendrogram(sch.linkage(x_pca,method='ward'))
hc = AgglomerativeClustering(n_clusters=18,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(x_pca)


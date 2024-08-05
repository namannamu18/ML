import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

X = pd.read_csv("synthetic_data.csv")


x1 = X['Feature1'].values
x2 = X['Feature2'].values
X = np.array(list(zip(x1, x2)))

plt.figure(figsize=(8, 6))
plt.title('Dataset')
plt.scatter(x1, x2, c='blue', marker='o', edgecolor='k', s=50)
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.show()

gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(X)
em_predictions = gmm.predict(X)


print("\nEM predictions")
print(em_predictions)
print("Means:\n", gmm.means_)
print("\nCovariances:\n", gmm.covariances_)


plt.figure(figsize=(8, 6))
plt.title('Expectation-Maximization (Gaussian Mixture)')
plt.scatter(X[:, 0], X[:, 1], c=em_predictions, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.show()


kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
kmeans_labels = kmeans.labels_
kmeans_centers = kmeans.cluster_centers_


print("\nk-Means cluster centers:")
print(kmeans_centers)


plt.figure(figsize=(8, 6))
plt.title('k-Means Clustering')
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='rainbow', marker='o', edgecolor='k', s=50)
plt.scatter(kmeans_centers[:, 0], kmeans_centers[:, 1], color='black', marker='X', s=200, label='Centroids')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.show()

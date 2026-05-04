import matplotlib.pyplot as plt
import numpy as np

# Dataset
X = np.array([
    [95,65],[88,60],[102,68],[90,62],[85,55],
    [60,30],[65,35],[55,25],[70,40],[58,28]
])

# Initial centroids
centroids = np.array([
    [95,65],
    [60,30]
])

clusters = []

for point in X:
    d1 = np.linalg.norm(point - centroids[0])
    d2 = np.linalg.norm(point - centroids[1])
    clusters.append(0 if d1 < d2 else 1)

plt.figure()

for i, point in enumerate(X):
    if clusters[i] == 0:
        plt.scatter(point[0], point[1])
    else:
        plt.scatter(point[0], point[1], marker='x')

# Plot centroids
plt.scatter(centroids[:,0], centroids[:,1], marker='*', s=200)

plt.xlabel("ADR")
plt.ylabel("Headshot Percentage")
plt.title("K-Means Initial Clustering")

plt.savefig("kmeans_clusters.png")
plt.show()
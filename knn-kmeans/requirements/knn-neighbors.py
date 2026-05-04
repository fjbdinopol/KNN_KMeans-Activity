import matplotlib.pyplot as plt
import math

data = [
    (95,65,1),(88,60,1),(102,68,1),(90,62,1),(85,55,1),
    (60,30,0),(65,35,0),(55,25,0),(70,40,0),(58,28,0)
]

new_point = (92,64)

# Compute distances
distances = []
for point in data:
    d = math.sqrt((92-point[0])**2 + (64-point[1])**2)
    distances.append((d, point))

# Sort
distances.sort()

# Get K = 3
neighbors = distances[:3]

plt.figure()

# Plot all points
for x,y,r in data:
    if r == 1:
        plt.scatter(x,y)
    else:
        plt.scatter(x,y, marker='x')

# Highlight neighbors
for d, (x,y,r) in neighbors:
    plt.scatter(x,y, s=200, edgecolors='black', facecolors='none')

# Plot new point
plt.scatter(92,64, marker='*', s=200)

plt.xlabel("ADR")
plt.ylabel("Headshot Percentage")
plt.title("KNN Nearest Neighbors (K=3)")

plt.savefig("knn_neighbors.png")
plt.show()
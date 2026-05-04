import matplotlib.pyplot as plt
import math

data = [
    (95,65),(88,60),(102,68),(90,62),(85,55),
    (60,30),(65,35),(55,25),(70,40),(58,28)
]

new_point = (92,64)

distances = []
labels = []

for i, (x,y) in enumerate(data):
    d = math.sqrt((92-x)**2 + (64-y)**2)
    distances.append(d)
    labels.append(f"P{i+1}")

plt.figure()
plt.bar(labels, distances)

plt.xlabel("Players")
plt.ylabel("Distance")
plt.title("Distance from New Player (KNN)")

plt.savefig("knn_distances.png")
plt.show()
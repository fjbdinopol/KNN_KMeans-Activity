import matplotlib.pyplot as plt

# Dataset
adr = [95,88,102,90,85,60,65,55,70,58]
hs = [65,60,68,62,55,30,35,25,40,28]
role = [1,1,1,1,1,0,0,0,0,0]

# Separate classes
adr_1 = [adr[i] for i in range(len(role)) if role[i] == 1]
hs_1 = [hs[i] for i in range(len(role)) if role[i] == 1]

adr_0 = [adr[i] for i in range(len(role)) if role[i] == 0]
hs_0 = [hs[i] for i in range(len(role)) if role[i] == 0]

# Plot
plt.figure()
plt.scatter(adr_1, hs_1, label="Aggressive (1)")
plt.scatter(adr_0, hs_0, marker='x', label="Passive (0)")

# New player
plt.scatter(92, 64, marker='*', s=200, label="New Player")

plt.xlabel("ADR")
plt.ylabel("Headshot Percentage")
plt.title("KNN Scatter Plot")
plt.legend()

plt.savefig("knn_scatter.png")
plt.show()
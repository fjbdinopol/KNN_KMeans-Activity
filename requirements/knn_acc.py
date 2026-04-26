import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# Load and Scale
df = pd.read_csv('files/diabetes-k-nn.csv')
scaler = MinMaxScaler()
X = scaler.fit_transform(df[['Glucose', 'BMI']]) # Using 2 features for 2D plot
y = df['Outcome'].values

# Create Meshgrid
h = .05
x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

fig, ax = plt.subplots(figsize=(8, 6))

def update(k):
    ax.clear()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20, cmap='coolwarm')
    ax.set_title(f"KNN Decision Boundary (K = {k})")
    ax.set_xlabel("Scaled Glucose")
    ax.set_ylabel("Scaled BMI")

# Frames: K values from 1 to 15
ks = [1, 3, 5, 7, 9, 11, 13, 15]
ani = animation.FuncAnimation(fig, update, frames=ks, interval=800)
ani.save('knn_boundaries.gif', writer='pillow')
plt.show()
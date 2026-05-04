import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# 1. Load and Clean Data
df = pd.read_csv('files/diabetes-k-nn.csv')
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols] = df[cols].replace(0, np.nan).fillna(df[cols].median())

# 2. Scale and Split
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df.drop('Outcome', axis=1))
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Outcome'], test_size=0.2, random_state=42)

# 3. Calculate Logistic Regression Baseline
lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))

# 4. Pre-calculate KNN Accuracies (Fixes the ValueError)
k_range = list(range(1, 21, 2)) # K = 1, 3, 5... 19
all_accuracies = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    all_accuracies.append(accuracy_score(y_test, knn.predict(X_test)))

# 5. Build the Animation
fig1, ax1 = plt.subplots(figsize=(8, 5))

def update_acc(i): # 'i' is the frame index
    ax1.clear()
    # Plot from the start up to the current frame
    ax1.plot(k_range[:i+1], all_accuracies[:i+1], marker='o', color='blue', label='KNN Accuracy')
    ax1.axhline(y=lr_acc, color='red', linestyle='--', label=f'LogReg Baseline ({lr_acc:.2%})')
    
    ax1.set_title(f"Model Performance Evolution (Testing up to K={k_range[i]})")
    ax1.set_xlabel("Number of Neighbors (K)")
    ax1.set_ylabel("Accuracy Score")
    ax1.set_ylim(0.6, 0.85)
    ax1.legend(loc='lower right')

ani1 = animation.FuncAnimation(fig1, update_acc, frames=len(k_range), interval=500, repeat=False)

# Save and Show
print("Generating Accuracy GIF...")
ani1.save('accuracy_evolution.gif', writer='pillow')
print("Done! Saved as accuracy_evolution.gif")
plt.show()
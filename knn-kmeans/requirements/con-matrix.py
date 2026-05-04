import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# 1. Load and Clean Data
df = pd.read_csv('files/diabetes-k-nn.csv')
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols] = df[cols].replace(0, np.nan).fillna(df[cols].median())

# 2. Scale and Split
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df.drop('Outcome', axis=1))
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Outcome'], test_size=0.2, random_state=42)

# 3. Define the K-values we want to showcase
cm_k_values = [3, 5, 7, 9, 11]

# 4. Build the Animation
fig2, ax2 = plt.subplots(figsize=(6, 4))

def update_cm(k):
    ax2.clear()
    
    # Train and evaluate for the current K
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    cm = confusion_matrix(y_test, knn.predict(X_test))
    
    # Draw the seaborn heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2,
                xticklabels=['Pred 0', 'Pred 1'], yticklabels=['Actual 0', 'Actual 1'])
    ax2.set_title(f"Confusion Matrix for K = {k}")

ani2 = animation.FuncAnimation(fig2, update_cm, frames=cm_k_values, interval=1000)

# Save and Show
print("Generating Confusion Matrix GIF...")
ani2.save('confusion_matrix_animated.gif', writer='pillow')
print("Done! Saved as confusion_matrix_animated.gif")
plt.show()
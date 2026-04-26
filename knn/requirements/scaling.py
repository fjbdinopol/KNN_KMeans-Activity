import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import MinMaxScaler

# --- 1. Load and Prepare the Data ---
try:
    df = pd.read_csv('files/diabetes-k-nn.csv')
except FileNotFoundError:
    print("Error: 'diabetes-k-nn.csv' not found. Ensure it is in the same folder.")
    exit()

# Handle missing zeroes using median imputation as discussed
cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_impute] = df[cols_to_impute].replace(0, np.nan)
df[cols_to_impute] = df[cols_to_impute].fillna(df[cols_to_impute].median())

# Separate the target and prepare unscaled data
X_unscaled = df.drop('Outcome', axis=1)

# Generate the scaled data
scaler = MinMaxScaler()
X_scaled_array = scaler.fit_transform(X_unscaled)
X_scaled = pd.DataFrame(X_scaled_array, columns=X_unscaled.columns)

# --- 2. Set Up the Animation Scene ---
fig, ax = plt.subplots(figsize=(10, 6))

# Define the animation function
def update(frame):
    # 'frame' goes from 0 to 60. Frame 60 represents 100% scaled.
    ax.clear()
    
    # Calculate the blend (alpha) between unscaled and scaled states
    alpha = frame / 60 
    
    # Linearly interpolate between the two DataFrames
    current_data = (1 - alpha) * X_unscaled + alpha * X_scaled
    
    # Create the boxplot for the current interpolated state
    current_data.boxplot(vert=False, ax=ax)
    
    # The fix: Dynamic X-axis limit adjustment
    # As alpha progresses, the x-axis limit contracts dynamically.
    ax.set_xlim(-10, 200 * (1 - alpha) + 1.1 * alpha)
    
    # Adaptive Title and X-axis Label
    percent_scaled = int(alpha * 100)
    ax.set_title(f"Normalization Progress: {percent_scaled}% Standardized", fontsize=14)
    ax.set_xlabel(f"{'Unscaled' if alpha==0 else 'Scaled'} Value Range", fontsize=12)

# --- 3. Build and Save the Animation ---
# Frames=61 creates a smooth effect. Increase interval (ms) for slower speed.
# 60 frames * 150ms = 9 seconds total animation time
ani = animation.FuncAnimation(fig, update, frames=range(61), interval=150, repeat=False)

# Save as GIF (requires 'pillow' library)
print("Generating and saving the animated GIF...")
try:
    ani.save('scaling_transition_fixed.gif', writer='pillow')
    print("Success! File saved as 'scaling_transition_fixed.gif'")
except Exception as e:
    print(f"Error saving animation: {e}\nEnsure 'pillow' is installed (pip install pillow).")

plt.show()
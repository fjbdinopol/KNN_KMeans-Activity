import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset
# Ensure 'diabetes-k-nn.csv' is in the same folder as this script
df = pd.read_csv('files/diabetes-k-nn.csv')

# GRAPH 1: Feature Correlation Heatmap
# Demonstrates which features have the strongest relationship with the Outcome 
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Part 1: Feature Correlation Heatmap')
plt.show() 

# GRAPH 2: Glucose Distribution
# Provides visual evidence of glucose as a primary predictor 
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Glucose', hue='Outcome', kde=True, palette='Set1', bins=30)
plt.title('Part 1: Glucose Distribution by Outcome')
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
iris = pd.read_csv("iris.csv")

# Extracting features and ground truth labels
X = iris.iloc[:, :-1].values  
y_true = iris.iloc[:, -1].values  

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert species labels to numerical values
species_mapping = {species: idx for idx, species in enumerate(np.unique(y_true))}
y_true_num = np.array([species_mapping[label] for label in y_true])

# Splitting dataset into 60% training and 40% testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_true_num, test_size=0.4, random_state=42, stratify=y_true_num)

# Function to map K-Means clusters to actual labels
def map_clusters_to_labels(y_true, y_kmeans):
    mapping = {}
    for cluster in np.unique(y_kmeans):
        mode_label = mode(y_true[y_kmeans == cluster], keepdims=True).mode[0]
        mapping[cluster] = mode_label
    return np.array([mapping[label] for label in y_kmeans])

# User input for k-value (number of clusters)
k_value = int(input("Enter the number of clusters (k): "))

# Apply K-Means clustering
kmeans = KMeans(n_clusters=k_value, init='k-means++', max_iter=300, n_init="auto", random_state=0)
y_kmeans = kmeans.fit_predict(X_test)

# Map clusters to ground truth labels
y_kmeans_mapped = map_clusters_to_labels(y_test, y_kmeans)

# Calculate Clustering Accuracy
accuracy = accuracy_score(y_test, y_kmeans_mapped)


# Print results
print(f"\nResults for k = {k_value}:")
print(f"K-Means Clustering Accuracy: {accuracy * 100:.2f}%")

# Visualizing Clusters (2D Projection)
plt.figure(figsize=(8, 6))
colors = ['purple', 'orange', 'green', 'blue', 'pink', 'yellow', 'cyan']
for i in range(k_value):
    plt.scatter(X_test[y_kmeans == i, 0], X_test[y_kmeans == i, 1], s=100, c=colors[i % len(colors)], label=f'Cluster {i+1}')

# Plotting centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.legend()
plt.title(f'K-Means Clustering (k={k_value})')
plt.show()

# 3D Visualization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

for i in range(k_value):
    ax.scatter(X_test[y_kmeans == i, 0], X_test[y_kmeans == i, 1], X_test[y_kmeans == i, 2], c=colors[i % len(colors)], label=f'Cluster {i+1}')

# Plot centroids
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], 
           s=200, c='red', marker='X', label='Centroids')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title(f'3D Visualization of Clusters (k={k_value})')
plt.legend()
plt.show()

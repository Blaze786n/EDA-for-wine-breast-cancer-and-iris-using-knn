import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split  # Importing train_test_split
from sklearn.decomposition import PCA

# Load dataset
file_path = "breast_cancer.csv"  # Change this to your actual file path
df = pd.read_csv("b_cancer.csv")

# Drop ID column (not useful for clustering)
df.drop(columns=['id'], inplace=True)

# Convert diagnosis column (M = 1, B = 0)
df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])

# Extract features (X) and target labels (y)
X = df.drop(columns=['diagnosis']).values
y_true = df['diagnosis'].values  # 0 = Benign, 1 = Malignant

# Standardizing features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into 60% training and 40% testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_true, test_size=0.40, random_state=42)

# Function to map K-Means clusters to actual labels
def map_clusters_to_labels(y_true, y_kmeans):
    mapping = {}
    for cluster in np.unique(y_kmeans):
        mode_label = mode(y_true[y_kmeans == cluster], keepdims=True).mode[0]
        mapping[cluster] = mode_label
    return np.array([mapping[label] for label in y_kmeans])

# User input for number of clusters (k)
k_value = int(input("Enter the number of clusters (k): "))

# Apply K-Means clustering on the training set
kmeans = KMeans(n_clusters=k_value, init='k-means++', max_iter=300, n_init="auto", random_state=0)
y_kmeans_train = kmeans.fit_predict(X_train)

# Map clusters to original labels on the training set
y_kmeans_train_mapped = map_clusters_to_labels(y_train, y_kmeans_train)

# Apply K-Means clustering on the test set (use the same fitted kmeans model)
y_kmeans_test = kmeans.predict(X_test)

# Map clusters to original labels on the test set
y_kmeans_test_mapped = map_clusters_to_labels(y_test, y_kmeans_test)

# Calculate Accuracy on test set
accuracy = accuracy_score(y_test, y_kmeans_test_mapped)

# Print performance metrics
print(f"\nResults for k = {k_value}:")
print(f"K-Means Accuracy on Test Set: {accuracy * 100:.2f}%")

# Visualizing Clusters (2D Projection using PCA)
# Perform PCA on the training set
pca = PCA(n_components=2)
X_pca_train = pca.fit_transform(X_train)  # Use only training data for PCA

plt.figure(figsize=(8, 6))
colors = ['purple', 'orange', 'green', 'blue', 'pink']
for i in range(k_value):
    plt.scatter(X_pca_train[y_kmeans_train == i, 0], X_pca_train[y_kmeans_train == i, 1], s=100, c=colors[i % len(colors)], label=f'Cluster {i+1}')

# Plot centroids (on the PCA transformed space of the training data)
centroids_pca = pca.transform(kmeans.cluster_centers_)  # Transform centroids using the same PCA
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=200, c='red', marker='X', label='Centroids')

plt.legend()
plt.title(f'K-Means Clustering (k={k_value}) - 2D PCA')
plt.show()

# 3D Visualization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

for i in range(k_value):
    ax.scatter(X_train[y_kmeans_train == i, 0], X_train[y_kmeans_train == i, 1], X_train[y_kmeans_train == i, 2], 
               c=colors[i % len(colors)], label=f'Cluster {i+1}')

# Plot centroids
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], 
           s=200, c='red', marker='X', label='Centroids')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title(f'3D Visualization of Clusters (k={k_value})')
plt.legend()
plt.show()

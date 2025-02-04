import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def silhouette_score(X, labels):

    n_samples = X.shape[0]
    sil_score = 0
    
    for i in range(n_samples):
        
        label_i = labels[i]
        
        # Calculate intra-cluster distance (a(i))
        same_cluster_points = X[labels == label_i]
        a_i = np.mean(pairwise_distances([X[i]], same_cluster_points))  # distance to all other points in the same cluster
        
        # Calculate nearest-cluster distance (b(i))
        other_labels = np.unique(labels[labels != label_i])
        b_i = np.inf  # initialize with a large value
        
        for label in other_labels:
            other_cluster_points = X[labels == label]
            b_i = min(b_i, np.mean(pairwise_distances([X[i]], other_cluster_points)))  # distance to the nearest cluster
        
        # Calculate the silhouette score for this point
        sil_score += (b_i - a_i) / max(a_i, b_i)
    
    # Calculate the average silhouette score for the entire dataset
    return sil_score / n_samples

def visualize_clusters(pca_features, cluster_labels):
 
    # Use the first two PCA components for visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(
        pca_features[:, 0],  # First PCA component
        pca_features[:, 1],  # Second PCA component
        c=cluster_labels,    # Cluster labels
        cmap='viridis',      # Color map for clusters
        alpha=1            # Transparency for better visibility
    )

    plt.colorbar(label="Cluster")
    plt.title("Cluster Visualization (Using PCA Components)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.show()


def visualize_clusters_3d(X, labels):
  
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis',alpha=1)
    
    # Add color bar
    plt.colorbar(sc)
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.set_title('3D Cluster Visualization')
    
    plt.show()

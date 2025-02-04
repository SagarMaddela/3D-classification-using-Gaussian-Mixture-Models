import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def find_optimal_clusters_bic(data, max_clusters=10):
    """
    Determine the optimal number of clusters using BIC with GMM.

    Args:
        data (np.array): Preprocessed data.
        max_clusters (int): Maximum number of clusters to test.

    Returns:
        int: Optimal number of clusters based on BIC.
    """
    bic_scores = []

    for k in range(1, max_clusters + 1):
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(data)
        bic_scores.append(gmm.bic(data))

    # Find the optimal number of clusters where BIC is minimized
    optimal_clusters = np.argmin(bic_scores) + 1

    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), bic_scores, marker='o', linestyle='--')
    plt.title("BIC Scores for Different Numbers of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("BIC Score")
    plt.grid(True)
    plt.show()

    print(f"[INFO] Optimal number of clusters based on BIC: {optimal_clusters}")
    return optimal_clusters

class CustomPCA:
    def __init__(self, n_components):
        """
        Custom PCA implementation.

        Args:
            n_components (int): Number of principal components to retain.
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, data):
        """
        Fit the PCA model to the data.

        Args:
            data (np.array): Input data of shape (n_samples, n_features).
        """
        # Step 1: Center the data
        self.mean = np.mean(data, axis=0)
        centered_data = data - self.mean

        # Step 2: Compute the covariance matrix
        covariance_matrix = np.cov(centered_data, rowvar=False)

        # Step 3: Perform eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Step 4: Sort eigenvectors by descending eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Step 5: Select the top 'n_components' eigenvectors
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]

    def transform(self, data):
        """
        Transform the data to the reduced-dimensional space.

        Args:
            data (np.array): Input data of shape (n_samples, n_features).

        Returns:
            np.array: Reduced data of shape (n_samples, n_components).
        """
        if self.components is None:
            raise ValueError("The PCA model must be fitted before transforming data.")
        centered_data = data - self.mean
        return np.dot(centered_data, self.components)

    def fit_transform(self, data):
        """
        Fit the PCA model and transform the data in one step.

        Args:
            data (np.array): Input data of shape (n_samples, n_features).

        Returns:
            np.array: Reduced data of shape (n_samples, n_components).
        """
        self.fit(data)
        return self.transform(data)


def train_model(data, n_components=5, max_clusters=10):
    """
    Train the clustering model using PCA for dimensionality reduction and GMM for clustering.

    Args:
        data (np.array): Preprocessed data.
        n_components (int): Number of PCA components.
        max_clusters (int): Maximum number of clusters to test for BIC.

    Returns:
        tuple: (trained GMM model, trained PCA model, cluster labels)
    """
    # Step 1: Perform PCA for dimensionality reduction
    print("[INFO] Performing PCA...")
    pca_model = CustomPCA(n_components=n_components)
    pca_features = pca_model.fit_transform(data)
    #print(f"[INFO] PCA completed. Explained variance ratio: {pca_model.explained_variance_ratio_}")

    # Step 2: Determine the optimal number of clusters using BIC
    print("[INFO] Determining the optimal number of clusters using BIC...")
    n_clusters = find_optimal_clusters_bic(pca_features, max_clusters)

    # Step 3: Train GMM with the optimal number of clusters
    print(f"[INFO] Training GMM with {n_clusters} clusters...")
    gmm_model = GaussianMixture(n_components=n_clusters, random_state=42)
    cluster_labels = gmm_model.fit_predict(pca_features)
    print("[INFO] GMM training complete.")

    return gmm_model, pca_model, cluster_labels

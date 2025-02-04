from data_processing import preprocess_data
from train import train_model
from evaluate import visualize_clusters
from evaluate import silhouette_score
from evaluate import visualize_clusters_3d

def main():
    # Step 1: Load and preprocess data
    file_path = 'dataset.csv'  
    X_train, X_test = preprocess_data(file_path)
    
    # Step 2: Train the clustering model using PCA and GMM
    n_components = 3  
    n_clusters = 10   
    gmm_model, pca_model, _ = train_model(X_train, n_components, n_clusters)
    
    # Step 3: Use the PCA model to transform both train and test data
    print("[DEBUG] Transforming training data using PCA...")
    pca_features_train = pca_model.transform(X_train)  # Transform the train data
    print(f"[DEBUG] PCA features shape for train data: {pca_features_train.shape}")
    
    print("[DEBUG] Transforming test data using PCA...")
    pca_features_test = pca_model.transform(X_test)  
    print(f"[DEBUG] PCA features shape for test data: {pca_features_test.shape}")
    
    print("[DEBUG] Predicting cluster labels for train data using GMM...")
    cluster_labels_train = gmm_model.predict(pca_features_train)  # Predict cluster labels for the train set
    
    print("[DEBUG] Predicting cluster labels for test data using GMM...")
    cluster_labels_test = gmm_model.predict(pca_features_test)  
    
    # Step 5: Visualize clusters 
    print("[DEBUG] Visualizing clusters for train dataset...")
    visualize_clusters(pca_features_train, cluster_labels_train)
    if n_components == 3:
        print("[DEBUG] Visualizing clusters for train dataset in 3D...")
        visualize_clusters_3d(pca_features_train, cluster_labels_train)
    
    print("[DEBUG] Visualizing clusters for test dataset...")
    visualize_clusters(pca_features_test, cluster_labels_test)
    if n_components == 3:
        print("[DEBUG] Visualizing clusters for test dataset in 3D...")
        visualize_clusters_3d(pca_features_test, cluster_labels_test)

    
    # Step 6: Calculate Silhouette Scores for both train and test clusters
    print("[DEBUG] Calculating Silhouette Score for train data...")
    silhouette_train = silhouette_score(pca_features_train, cluster_labels_train)
    print(f"[DEBUG] Silhouette Score for Train Data: {silhouette_train:.4f}")
    
    print("[DEBUG] Calculating Silhouette Score for test data...")
    silhouette_test = silhouette_score(pca_features_test, cluster_labels_test)
    print(f"[DEBUG] Silhouette Score for Test Data: {silhouette_test:.4f}")

if __name__ == "__main__":
    main()

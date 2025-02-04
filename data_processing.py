import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

def preprocess_data(file_path):

    try:
        # Step 1: Resolve file path
        print(f"[DEBUG] Dataset file path resolved: {file_path}")
        
        # Step 2: Load dataset
        print("[DEBUG] Loading dataset...")
        df = pd.read_csv(file_path)
        print(f"[DEBUG] Dataset loaded successfully with shape: {df.shape}")
        print(f"[DEBUG] Columns in dataset: {list(df.columns)}")
        
        # Step 3: Define features
        numeric_features = ['Age', 'Income']  # Use 'Age' and 'Income' as numeric features
        categorical_features = ['Sex', 'Marital status', 'Education', 'Occupation', 'Settlement size']  # Categorical features
        
        print(f"[DEBUG] Numeric features: {numeric_features}")
        print(f"[DEBUG] Categorical features: {categorical_features}")
        
        # Step 4: Preprocess the data
        print("[DEBUG] Starting preprocessing...")
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ]
        )
        
        data = preprocessor.fit_transform(df)
        print("[DEBUG] Preprocessing complete.")
        print(f"[DEBUG] Preprocessed data shape: {data.shape}")
        
        # Step 6: Split data into 80% train and 20% test (Hold-out method)
        test_size_ratio = 0.2 
        shuffle = True         

        if shuffle:
            np.random.seed(42)  # Set seed for reproducibility
            np.random.shuffle(data) 
            test_size = int(len(data) * test_size_ratio) 
        # Split the data
        X_train = data[:-test_size]
        X_test = data[-test_size:]   

        print(f"[DEBUG] Data split into train and test sets. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test
    
    except Exception as e:
        print(f"[ERROR] An error occurred during preprocessing: {e}")
        raise


# # Ensure this function is called
# if __name__ == "__main__":
#     print("[INFO] Starting data preprocessing...")
#     data = preprocess_data()
#     print("[INFO] Preprocessing finished successfully!")

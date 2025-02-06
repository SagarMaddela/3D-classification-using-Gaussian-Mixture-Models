# 3D Clustering using Gaussian Mixture Models (GMM)

## 📌 Project Overview

This project implements **customer segmentation** using **Principal Component Analysis (PCA)** for dimensionality reduction and **Gaussian Mixture Models (GMM)** for clustering. It analyzes customer data and visualizes the clusters in **2D and 3D**. The optimal number of clusters is determined using the **Bayesian Information Criterion (BIC)**.

## ✨ Features

- **Preprocessing:** Handles numerical and categorical data using **StandardScaler** and **OneHotEncoder**.
- **Dimensionality Reduction:** Implements a **custom PCA algorithm**.
- **Clustering:** Uses **Gaussian Mixture Models (GMM)** for clustering.
- **Model Evaluation:** Computes **Silhouette Scores** to assess clustering performance.
- **Visualization:** Displays clusters in **2D and 3D using Matplotlib**.

## 📂 Project Structure

```
📁 3D_GMM_Clustering
│── main.py               # Main script to execute the pipeline
│── train.py              # PCA and GMM training module
│── evaluate.py           # Clustering evaluation and visualization
│── data_processing.py    # Preprocessing functions for data
│── dataset.csv           # Input dataset
│── README.md             # Project documentation
│── requirements.txt      # Necessary packages for project 
```

## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SagarMaddela/3D-clustering-using-Gaussian-Mixture-Models.git
   cd 3D-clustering-using-Gaussian-Mixture-Models
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

Run the main script to execute the pipeline:

```bash
python main.py
```

The script will:

1. Load and preprocess the dataset (`dataset.csv`).
2. Apply **PCA for dimensionality reduction**.
3. Determine the **optimal number of clusters** using BIC.
4. Train a **GMM model** and predict clusters.
5. Compute **Silhouette Scores**.
6. **Visualize clusters** in 2D and 3D.

## 📊 Visualizations

The project provides two types of cluster visualizations:

- **2D Plot** (First two PCA components)
- **3D Scatter Plot** (If `n_components=3` in PCA)

## 🛠 Troubleshooting

- If **dataset.csv** is missing, ensure you place a valid dataset in the project folder.
- Ensure `matplotlib`, `numpy`, and `sklearn` are installed before running the project.
- Adjust the `n_components` and `max_clusters` parameters in `main.py` for experimentation.

## 📜 License

This project is open-source and available under the **MIT License**.



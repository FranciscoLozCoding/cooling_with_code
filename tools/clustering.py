"""This module contains clustering tools."""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.metrics import r2_score

def run_dbscan_clustering(df, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering on a dataset and visualize clusters.

    Parameters:
        df (pd.DataFrame): The dataset to cluster (numerical features only).
        eps (float): The maximum distance between two samples for one to be considered a neighbor.
        min_samples (int): The minimum number of points required to form a dense region.

    Returns:
        pd.DataFrame: The dataset with an added 'Cluster' column indicating cluster assignments.
    """
    # Select numerical features
    df_numeric = df.select_dtypes(include=[np.number])

    # Standardize features (DBSCAN is sensitive to feature scaling)
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

    # Run DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(df_scaled)

    # Add cluster labels to the dataset
    df['Cluster'] = clusters

    return df

def find_optimal_eps(df):
    """
    Plot the k-distance graph to determine the optimal epsilon for DBSCAN.

    Parameters:
        df (pd.DataFrame): The dataset (numerical features only).
    """
    df_numeric = df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)

    # Compute the k-nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(df_scaled)
    distances, indices = neighbors_fit.kneighbors(df_scaled)

    # Sort distances for visualization
    distances = np.sort(distances[:, -1], axis=0)

    # Plot k-distance graph
    plt.figure(figsize=(8, 5))
    plt.plot(distances)
    plt.xlabel("Points sorted by distance")
    plt.ylabel("5-NN distance")
    plt.title("Elbow Method for Choosing Epsilon (eps)")
    plt.show()

def plot_dbscan_pca(df, dataset_name, labels=None):
    """
    Reduce DBSCAN clusters to 2D using PCA for visualization.

    Parameters:
        df (pd.DataFrame): Clustered dataset with a 'Cluster' column.
        dataset_name (str): Name of dataset for labeling.
        labels (list): List of target variable labels.
    """
    # Drop non-numeric columns and Cluster column
    df_numeric = df.select_dtypes(include=[np.number]).drop(columns=['Cluster'], errors='ignore')

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_numeric)

    # Convert PCA results to a DataFrame
    df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = df.loc[df_numeric.index, 'Cluster'].values  # Match clusters to PCA results

    # Plot PCA results with cluster labels
    plt.figure(figsize=(10, 6))
    if labels:
        # Add target variable labels to PCA DataFrame
        df_pca['UHI_category'] = labels

        sns.scatterplot(
            x=df_pca['PC1'],
            y=df_pca['PC2'],
            hue=df_pca['Cluster'],
            style=df_pca['UHI_category'],
            palette="tab10",
            alpha=0.7
        )
    else:
      sns.scatterplot(
          x=df_pca['PC1'],
          y=df_pca['PC2'], 
          hue=df_pca['Cluster'],
          palette="tab10", 
          alpha=0.7
      )
    plt.title(f'DBSCAN Clustering in PCA Space ({dataset_name})')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()

def calculate_cluster_category_percentages(df, cluster_col='Cluster', category_col='UHI_category'):
    """
    Calculate the percentage of each UHI_category label within each DBSCAN Cluster.

    Parameters:
        df (pd.DataFrame): DataFrame containing cluster assignments and UHI categories.
        cluster_col (str): Column name for clusters (default: 'Cluster').
        category_col (str): Column name for UHI categories (default: 'UHI_category').

    Returns:
        pd.DataFrame: A DataFrame showing the percentage of each UHI category within each cluster.
    """
    # Count occurrences of each category in each cluster
    cluster_category_counts = df.groupby([cluster_col, category_col]).size().unstack(fill_value=0)

    # Normalize by row to get percentages
    cluster_category_percentages = cluster_category_counts.div(cluster_category_counts.sum(axis=1), axis=0) * 100

    return cluster_category_percentages

def run_gmm_clustering(df, dataset_name, n_components=3, random_state=42):
    """
    Runs Gaussian Mixture Model (GMM) clustering on the dataset.

    Parameters:
        df (pd.DataFrame): The dataset containing numerical features.
        dataset_name (str): Name of dataset for visualization purposes.
        n_components (int): Number of clusters to fit (default=3).
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Dataset with an added 'GMM_Cluster' column.
    """
    # Select only numeric features (excluding categorical labels)
    df_numeric = df.select_dtypes(include=[np.number]).drop(columns=['Cluster'], errors='ignore')

    # Standardize features (GMM is sensitive to scale)
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

    # Fit GMM model
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    df['GMM_Cluster'] = gmm.fit_predict(df_scaled)  # Assign clusters

    return df

def find_optimal_gmm_clusters(df, max_clusters=10):
    """
    Determines the optimal number of clusters for GMM using BIC and AIC.

    Parameters:
        df (pd.DataFrame): Dataset with numerical features.
        max_clusters (int): Maximum number of clusters to test.

    Returns:
        None (Plots BIC and AIC scores for different cluster sizes).
    """
    df_numeric = df.select_dtypes(include=[np.number]).drop(columns=['Cluster'], errors='ignore')
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)

    bic_scores, aic_scores = [], []
    cluster_range = range(1, max_clusters + 1)

    for k in cluster_range:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(df_scaled)
        bic_scores.append(gmm.bic(df_scaled))
        aic_scores.append(gmm.aic(df_scaled))

    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, bic_scores, label='BIC', marker='o')
    plt.plot(cluster_range, aic_scores, label='AIC', marker='s')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('GMM Cluster Selection (BIC & AIC)')
    plt.legend()
    plt.show()

def plot_gmm_pca(df, dataset_name, labels=None):
    """
    Visualizes GMM clusters using PCA.

    Parameters:
        df (pd.DataFrame): Clustered dataset with 'GMM_Cluster' column.
        dataset_name (str): Name of dataset for labeling.
        labels (list): List of target variable labels.
    """
    # Select only numeric columns (excluding 'GMM_Cluster') and drop NaN values
    df_numeric = df.select_dtypes(include=[np.number]).drop(columns=['GMM_Cluster'], errors='ignore').dropna()

    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_numeric)

    # Convert PCA results to DataFrame
    df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    df_pca['GMM_Cluster'] = df.loc[df_numeric.index, 'GMM_Cluster'].values  # Match clusters to PCA results

    # Plot PCA results with cluster labels
    plt.figure(figsize=(10, 6))
    if labels:
      # Add target variable labels to PCA DataFrame
      df_pca['UHI_category'] = labels

      sns.scatterplot(
          x=df_pca['PC1'],
          y=df_pca['PC2'],
          hue=df_pca['GMM_Cluster'],
          style=df_pca['UHI_category'],
          palette="tab10",
          alpha=0.7
      )
    else:
      sns.scatterplot(
          x=df_pca['PC1'],
          y=df_pca['PC2'], 
          hue=df_pca['GMM_Cluster'],
          palette="tab10", 
          alpha=0.7
      )
    plt.title(f'GMM Clustering in PCA Space ({dataset_name})')
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="GMM Cluster")
    plt.tight_layout()
    plt.show()

def calculate_weighted_cluster_category_percentages(df, cluster_col='Cluster', category_col='UHI_category'):
    """
    Calculate the weighted percentage of each UHI_category label within each cluster,
    while ensuring final percentages sum to 100%.

    Parameters:
        df (pd.DataFrame): DataFrame containing cluster assignments and UHI categories.
        cluster_col (str): Column name for clusters (default: 'Cluster').
        category_col (str): Column name for UHI categories (default: 'UHI_category').

    Returns:
        pd.DataFrame: A DataFrame showing the weighted percentage of each UHI category per cluster,
                      with each row summing to 100%.
    """
    # Step 1: Get the total category distribution in the dataset
    total_category_counts = df[category_col].value_counts(normalize=True)  # Get proportions

    # Step 2: Count occurrences of each category in each cluster
    cluster_category_counts = df.groupby([cluster_col, category_col]).size().unstack(fill_value=0)

    # Step 3: Normalize by row (percentage of each category within a cluster)
    cluster_category_percentages = cluster_category_counts.div(cluster_category_counts.sum(axis=1), axis=0) * 100

    # Step 4: Adjust for category imbalances by dividing by the dataset-wide distribution
    weighted_cluster_percentages = cluster_category_percentages.div(total_category_counts, axis=1)

    # Step 5: Re-normalize so that each row sums to 100%
    weighted_cluster_percentages = weighted_cluster_percentages.div(weighted_cluster_percentages.sum(axis=1), axis=0) * 100

    return weighted_cluster_percentages

def evaluate_clustering_r2(df, cluster_col, category_col, cluster_mapping, ignore_clusters=None):
    """
    Generalized function to evaluate how well clustering results align with UHI categories using R².

    Parameters:
        df (pd.DataFrame): The dataset containing cluster assignments and UHI categories.
        cluster_col (str): Column name containing cluster labels (e.g., 'GMM_Cluster', 'DBSCAN_Cluster').
        category_col (str): Column name containing true UHI category labels.
        cluster_mapping (dict): Mapping of cluster numbers to UHI labels (e.g., {0: 'same_as_mean', 2: 'cooler', 3: 'hotter'}).
        ignore_clusters (list, optional): List of clusters to ignore (e.g., outliers in DBSCAN, noise).

    Returns:
        float: R² score representing how well clusters match UHI labels.
    """
    # Copy data to avoid modifying the original DataFrame
    filtered_df = df.copy()

    # Remove ignored clusters (if specified)
    if ignore_clusters:
        filtered_df = filtered_df[~filtered_df[cluster_col].isin(ignore_clusters)]

    # Map clusters to categorical labels
    filtered_df['Predicted_UHI'] = filtered_df[cluster_col].map(cluster_mapping)

    # Convert categorical labels to numerical values
    category_mapping = {'cooler': 0, 'same_as_mean': 1, 'hotter': 2}
    filtered_df['True_UHI'] = filtered_df[category_col].map(category_mapping)
    filtered_df['Predicted_UHI'] = filtered_df['Predicted_UHI'].map(category_mapping)

    # Drop rows where mapping failed (e.g., NaN due to unrecognized clusters)
    filtered_df = filtered_df.dropna()

    # Compute R² score
    r2 = r2_score(filtered_df['True_UHI'], filtered_df['Predicted_UHI'])

    return r2
"""This module includes outlier tools for the project."""
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

def univariate_outlier_analysis(df, dataset_name):
    """
    This function conducts a univariate outlier analysis on the DataFrame passed in.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
        dataset_name (str): The name of the dataset.

    Returns:
        None
    """
    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    outlier_df = scaler.fit_transform(df)

    # Create the box plot
    plt.figure(figsize=(20, 10))
    sns.boxplot(data=outlier_df)
    plt.xticks(ticks=range(len(df.columns)), labels=df.columns, rotation=45, ha='right') # Rotate x-axis labels for readability
    plt.tight_layout()
    plt.title(f'{dataset_name} Univariate Outlier Analysis')
    plt.show()

def multivariate_outlier_analysis(df, dataset_name, contamination=0.05, random_state=42):
    """
    Perform multivariate outlier analysis using Isolation Forest.

    Parameters:
        df (pd.DataFrame): The DataFrame containing numerical features.
        dataset_name (str): Name of the dataset for visualization purposes.
        contamination (float): Expected proportion of outliers in the data (default 5%).
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Original DataFrame with an additional column "Outlier" (1 = outlier, 0 = normal).
    """
    # Copy the dataset to avoid modifying the original
    df_copy = df.copy()
    
    # Scale the features for better Isolation Forest performance
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_copy), columns=df_copy.columns)
    
    # Apply Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    df_copy['Outlier'] = iso_forest.fit_predict(df_scaled)
    
    # Map: -1 (outliers) to 1,  1 (normal) to 0 for easier interpretation
    df_copy['Outlier'] = df_copy['Outlier'].map({1: 0, -1: 1})
    
    # Show percentage of outliers
    outlier_count = df_copy['Outlier'].sum()
    print(f"Detected {outlier_count} outliers ({(outlier_count / len(df_copy) * 100):.2f}% of total) in {dataset_name}")
    
    return df_copy

def plot_multivariate_outliers(df, dataset_name, n_components=2):
    """
    Perform PCA on high-dimensional data and plot outliers in 2D.
    
    Parameters:
        df (pd.DataFrame): DataFrame with an "Outlier" column (0 = normal, 1 = outlier).
        dataset_name (str): Name of the dataset for visualization purposes.
        n_components (int): Number of PCA components (default=2).
    """
    # Drop non-numeric columns and outlier column
    df_numeric = df.select_dtypes(include=[np.number]).drop(columns=['Outlier'], errors='ignore')

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df_numeric)

    # Create a DataFrame with PCA results
    df_pca = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
    df_pca['Outlier'] = df['Outlier'].values

    # Scatter plot of PCA components
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=df_pca['PC1'], 
        y=df_pca['PC2'], 
        hue=df_pca['Outlier'], 
        palette={0: "blue", 1: "red"},
        alpha=0.7
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f'PCA Visualization of {dataset_name} with Outliers')
    
    # Create custom legend handles.
    handles = [mpatches.Patch(color="blue", label="Normal"),
               mpatches.Patch(color="red", label="Outlier")]
    plt.legend(handles=handles, title="Outlier", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()
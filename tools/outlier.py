"""Outlier tools for the project."""
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

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
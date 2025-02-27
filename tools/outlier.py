"""Outlier tools for the project."""
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

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
"""Distribution tools for the project."""
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

def plot_target_var_distribution(orginal_dataset, train_split, valid_split):
    """
    Plot distribution of target variable in original, training, and validation datasets.

    Parameters
    ----------
    orginal_dataset : array-like
        Array of the original dataset.
    train_split : array-like
        Array of the training dataset.
    valid_split : array-like
        Array of the validation dataset.

    Returns
    -------
    plt : matplotlib.pyplot
        Matplotlib plot object.
    """
    plt.figure(figsize=(12, 6))
    # Plot histogram for the original training dataset.
    plt.hist(orginal_dataset, bins=30, alpha=0.5, label='Original Dataset', color='blue')

    # Plot histogram for the training split.
    plt.hist(train_split, bins=30, alpha=0.5, label='Training Split', color='green')

    # Plot histogram for the validation split.
    plt.hist(valid_split, bins=30, alpha=0.5, label='Validation Split', color='red')

    plt.xlabel('UHI')
    plt.ylabel('Frequency')
    plt.title('Distribution of UHI: Original, Training, and Validation Splits')
    plt.legend()
    return plt

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
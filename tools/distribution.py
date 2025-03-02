"""This module contains tools for analyzing feature distributions."""
import matplotlib.pyplot as plt

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

def plot_all_histograms(df, dataset_name, bins=50):
    """
    Plot histograms for all numerical features in a dataset.
    
    Parameters:
        df (pd.DataFrame): The dataset containing numerical features.
        dataset_name (str): Name of the dataset for the title.
        bins (int): Number of bins for histograms.
    """
    num_cols = df.select_dtypes(include=['number']).columns  # Select only numerical columns
    num_features = len(num_cols)
    
    # Define subplot grid size (auto-adjusting)
    rows = (num_features // 4) + 1  # 4 histograms per row
    fig, axes = plt.subplots(rows, 4, figsize=(15, rows * 4))  # Dynamic size
    axes = axes.flatten()  # Flatten for easy iteration
    
    # Loop through each numerical column and plot histogram
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col], bins=bins, color="steelblue", alpha=0.7)
        axes[i].set_title(f"Histogram of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle(f"Histograms of Features in {dataset_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit suptitle
    return plt
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
"""This module contains tools for model improvement."""
import matplotlib.pyplot as plt
import seaborn as sns

def plot_actual_vs_predicted(y_true, y_pred, dataset_name):
    """Plots actual vs. predicted values with a regression line.

    Args:
        y_true: Array-like of true target values.
        y_pred: Array-like of predicted target values.
    """

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--') # Regression line

    plt.xlabel("Actual UHI")
    plt.ylabel("Predicted UHI")
    plt.title(f"{dataset_name} Actual vs. Predicted UHI Values")
    plt.grid(True)
    plt.show()

def plot_hist_residuals(y_true, y_pred, dataset_name):
    """
    Plots histogram of residuals.

    Args:
        y_true: Array-like of true target values.
        y_pred: Array-like of predicted target values.
        dataset_name: Name of the dataset.
    """
    # Compute residuals
    residuals = y_true - y_pred

    # Plot histogram
    plt.figure(figsize=(8,5))
    sns.histplot(residuals, bins=50, kde=True, color="blue")
    plt.axvline(x=0, color='red', linestyle='--', label='Zero Residual Line')
    plt.xlabel("Residual (Actual - Predicted UHI)")
    plt.ylabel("Frequency")
    plt.title(f"{dataset_name} Histogram of Residuals")
    plt.legend()
    plt.show()

def plot_residuals_vs_predicted(y_true, y_pred, dataset_name):
    """
    Plots residuals vs. predicted values.

    Args:
        y_true: Array-like of true target values.
        y_pred: Array-like of predicted target values.
        dataset_name: Name of the dataset.
    """
    # Compute residuals
    residuals = y_true - y_pred

    # Plot residuals vs. predicted values
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--', label="Zero Residual Line")
    plt.xlabel("Predicted UHI")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(f"{dataset_name} Residuals vs. Predicted UHI")
    plt.legend()
    plt.show()

def plot_residuals_vs_actual(y_true, y_pred, dataset_name):
    """
    Plots residuals vs. actual values.

    Args:
        y_true: Array-like of true target values.
        y_pred: Array-like of predicted target
    """
    # Compute residuals
    residuals = y_true - y_pred

    # Plot residuals vs. actual values 
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=y_true, y=residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--', label="Zero Residual Line")
    plt.xlabel("Actual UHI")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(f"{dataset_name} Residuals vs. Actual UHI")
    plt.legend()
    plt.show()
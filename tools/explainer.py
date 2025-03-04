"""This module provides an explainer for the model."""

import shap
import matplotlib.pyplot as plt
import pandas as pd

def apply_shap(model, explainer_type, X, feature_names):
    """
    Applies SHAP to any trained model and generates visualizations.
    
    Parameters:
    - model: Trained model (e.g., RandomForestRegressor, XGBRegressor).
    - explainer_type: SHAP explainer class (e.g., shap.TreeExplainer, shap.KernelExplainer).
    - X: Data (must be a Pandas DataFrame) to compute SHAP values.
    - feature_names: List of feature names.

    Returns:
    - shap_values: SHAP values computed from the explainer.
    - explainer: SHAP explainer object.
        
    Example Usage:
    apply_shap(best_model, shap.TreeExplainer, X_valid_selected_df, selected_feature_names)
    """
    
    # Ensure X is a DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)

    # Initialize SHAP Explainer
    explainer = explainer_type(model)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_df)

    return shap_values, explainer

def shap_summary_plot(shap_values, X):
    """
    Generates a SHAP summary plot showing global feature importance.
    
    Parameters:
    - shap_values: SHAP values computed from an explainer.
    - X: Data (Pandas DataFrame) with feature names.
    
    Example Usage:
    shap_summary_plot(shap_values, X_valid_selected_df)
    """
    shap.summary_plot(shap_values, X)

def shap_bar_plot(shap_values, X):
    """
    Generates a bar plot showing overall feature importance rankings.
    
    Parameters:
    - shap_values: SHAP values computed from an explainer.
    - X: Data (Pandas DataFrame) with feature names.
    
    Example Usage:
    shap_bar_plot(shap_values, X_valid_selected_df)
    """
    shap.summary_plot(shap_values, X, plot_type="bar")

def shap_dependence_plot(feature_name, shap_values, X):
    """
    Generates a SHAP dependence plot for a given feature.

    Parameters:
    - feature_name: The name of the feature to plot.
    - shap_values: SHAP values computed from an explainer.
    - X: Data (Pandas DataFrame) with feature names.
    
    Example Usage:
    shap_dependence_plot("Building Height", shap_values, X_valid_selected_df)
    """
    shap.dependence_plot(feature_name, shap_values, X)

def shap_force_plot(explainer, shap_values, X, index=0):
    """
    Generates a SHAP force plot for a single prediction instance.
    
    Parameters:
    - explainer: SHAP explainer used to generate shap_values.
    - shap_values: SHAP values computed from an explainer.
    - X: Data (Pandas DataFrame) with feature names.
    - index (int): The index of the observation of interest.
    
    Example Usage:
    shap_force_plot(explainer, shap_values, X_valid_selected_df, instance_index=0)
    """
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[index,:], X.iloc[index,:])

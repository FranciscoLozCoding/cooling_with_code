"""This module provides an explainer for the model."""

import shap
import matplotlib.pyplot as plt
import pandas as pd

class Explainer:
    """
    A class for SHAP-based model explanation.
    
    Attributes:
    - model: Trained model (e.g., RandomForestRegressor, XGBRegressor).
    - explainer_type: SHAP explainer class (e.g., shap.TreeExplainer, shap.KernelExplainer).
    - X: Data (Pandas DataFrame) used to compute SHAP values.
    - feature_names: List of feature names.
    - explainer: SHAP explainer instance.
    - shap_values: Computed SHAP values.

    Methods:
    - apply_shap(): Computes SHAP values.
    - summary_plot(): Generates a SHAP summary plot.
    - bar_plot(): Generates a bar chart of feature importance.
    - dependence_plot(): Generates a dependence plot for a feature.
    - force_plot(): Generates a force plot for an individual prediction.
    """

    def __init__(self, model, explainer_type, X, feature_names):
        """
        Initializes the Explainer with a trained model, explainer type, and dataset.
        
        Parameters:
        - model: Trained model (e.g., RandomForestRegressor, XGBRegressor).
        - explainer_type: SHAP explainer class (e.g., shap.TreeExplainer, shap.KernelExplainer).
        - X: Data (Pandas DataFrame) used to compute SHAP values.
        - feature_names: List of feature names.
        """
        self.model = model
        self.explainer_type = explainer_type
        self.X = pd.DataFrame(X, columns=feature_names)  # Ensure DataFrame format
        self.feature_names = feature_names
        self.explainer = explainer_type(model)  # Initialize explainer
        self.shap_values = self.explainer.shap_values(self.X)  # Compute SHAP values
        shap.initjs()  # Initialize SHAP for Jupyter Notebook

    def summary_plot(self):
        """Generates a SHAP summary plot showing global feature importance."""
        shap.summary_plot(self.shap_values, self.X)

    def bar_plot(self):
        """Generates a bar plot showing overall feature importance rankings."""
        shap.summary_plot(self.shap_values, self.X, plot_type="bar")

    def dependence_plot(self, feature_name):
        """
        Generates a SHAP dependence plot for a given feature.
        
        Parameters:
        - feature_name (str): The feature to analyze.
        """
        shap.dependence_plot(feature_name, self.shap_values, self.X)

    def force_plot(self, index=0):
        """
        Generates a SHAP force plot for a single prediction instance.
        
        Parameters:
        - index (int): The index of the observation of interest.
        """
        return shap.force_plot(self.explainer.expected_value, self.shap_values[index, :], self.X.iloc[index, :])

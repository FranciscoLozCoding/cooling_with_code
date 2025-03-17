"""This module provides an explainer for the model."""

import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    - init_js(): Initializes SHAP for Jupyter Notebook.
    - reasoning(): Provides insights on why a record received a high or low UHI index.
    """

    def __init__(self, model, explainer_type, X, feature_names, ref_data=None, shap_values=None):
        """
        Initializes the Explainer with a trained model, explainer type, and dataset.
        
        Parameters:
        - model: Trained model (e.g., RandomForestRegressor, XGBRegressor).
        - explainer_type: SHAP explainer class (e.g., shap.TreeExplainer, shap.KernelExplainer).
        - X: Data (Pandas DataFrame) used to compute SHAP values.
        - feature_names: List of feature names.
        - ref_data (optional): The reference dataset (background dataset) is used by SHAP to estimate the expected output of the model
        - shap_values (optional): Precomputed SHAP values
        """
        self.model = model
        self.explainer_type = explainer_type
        self.X = np.array(X) if isinstance(X, pd.DataFrame) else X  # Ensure NumPy format
        if ref_data is not None:
            ref_data = np.array(ref_data) if isinstance(ref_data, pd.DataFrame) else ref_data # Ensure NumPy format
        self.feature_names = feature_names
        self.explainer = explainer_type(model, ref_data)  # Initialize explainer
        # Compute SHAP values
        if shap_values is not None:
            self.shap_values = shap_values
        else:
            self.shap_values = self.explainer.shap_values(self.X, check_additivity=False) if self.explainer_type == shap.DeepExplainer else self.explainer.shap_values(self.X)
        # Apply squeeze only if the array has three dimensions and the last dimension is 1
        if self.shap_values.ndim == 3 and self.shap_values.shape[-1] == 1:
            self.shap_values = np.squeeze(self.shap_values)
        self.init_js()  # Initialize SHAP for Jupyter Notebook

    def summary_plot(self):
        """Generates a SHAP summary plot showing global feature importance."""
        shap.summary_plot(self.shap_values, self.X, feature_names=self.feature_names)

    def bar_plot(self):
        """Generates a bar plot showing overall feature importance rankings."""
        shap.summary_plot(self.shap_values, self.X, feature_names=self.feature_names, plot_type="bar")

    def dependence_plot(self, feature_name):
        """
        Generates a SHAP dependence plot for a given feature.
        
        Parameters:
        - feature_name (str): The feature to analyze.
        """
        shap.dependence_plot(feature_name, self.shap_values, self.X, feature_names=self.feature_names)

    def force_plot(self, index=0):
        """
        Generates a SHAP force plot for a single prediction instance.
        
        Parameters:
        - index (int): The index of the observation of interest.
        """
        if self.explainer_type == shap.DeepExplainer:
            # Ensure expected_value is a single value (not tensor)
            expected_value = np.array(self.explainer.expected_value)
        else:
            expected_value = self.explainer.expected_value
        return shap.force_plot(expected_value, self.shap_values[index, :], self.X[index, :], feature_names=self.feature_names)
    
    def init_js(self):
        """Initializes SHAP for Jupyter Notebook."""
        shap.initjs()
        
    def reasoning(self, index=0, location=(None, None)):
        """
        Provides insights on why the record received a high or low UHI index.

        Parameters:
            index (int): The index of the observation of interest.
            location (tuple) (optional): The location of the record (long, lat).

        Returns:
            dict: The insights for the selected record.
        """

        # if self.explainer_type == shap.DeepExplainer:
        #     # Ensure expected_value is a single value (not tensor)
        #     expected_value = np.array(self.explainer.expected_value)[0]
        # else:
        #     expected_value = self.explainer.expected_value

        expected_value = self.explainer.expected_value

        # Validate record index
        if index >= len(self.shap_values) or index < 0:
            return {"error": "Invalid record index"}

        # Extract SHAP values for the specified record
        record_shap_values = self.shap_values[index]

        # Compute SHAP-based final prediction
        shap_final_prediction = expected_value + sum(record_shap_values)

        # Structure feature contributions
        feature_contributions = [
            {
                "feature": feature,
                "shap_value": value,
                "impact": "increase" if value > 0 else "decrease"
            }
            for feature, value in zip(self.feature_names, record_shap_values)
        ]

        # Create JSON structure
        shap_json = {
            "record_index": index,
            "longitude": location[0],
            "latitude": location[1],
            "base_value": expected_value,
            "shap_final_prediction": shap_final_prediction,  # SHAP-based predicted value
            "uhi_status": "Urban Heat Island" if shap_final_prediction > 1 else "Cooler Region",
            "feature_contributions": feature_contributions,
        }

        return shap_json

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import shap

class UhiPredictor:
    """
    Urban Heat Island Predictor Class that predicts new instances and explains the prediction using SHAP.
    
    INPUTS
    ---
    model_path: str - Path to the trained model file.
    scaler_path: str - Path to the standard scaler file.
    explainer_type: SHAP explainer class (e.g., shap.TreeExplainer, shap.KernelExplainer).
    ref_data: pd.DataFrame or np.array - Background dataset for SHAP explainer.
    feature_names: list - Feature names for SHAP analysis.
    """

    def __init__(self, model_path, scaler_path, explainer_type, ref_data, feature_names):
        """
        Initializes the UHI predictor with a trained model, scaler, and SHAP explainer.

        INPUTS
        ---
        model_path: str - Path to the model file.
        scaler_path: str - Path to the standard scaler file.
        explainer_type: SHAP explainer class (e.g., shap.TreeExplainer, shap.KernelExplainer).
        ref_data: pd.DataFrame or np.array - Background dataset for SHAP explainer.
        feature_names: list - Feature names for SHAP explanation.
        """
        # Load the model and scaler
        self.model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Ensure reference data is in NumPy format
        ref_data = np.array(ref_data) if isinstance(ref_data, pd.DataFrame) else ref_data  
        
        # Initialize SHAP explainer
        self.explainer_type = explainer_type
        self.explainer = self.explainer_type(self.model, ref_data)
        self.feature_names = feature_names

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input DataFrame to create new features for the model.

        INPUT
        -----
        df: pd.DataFrame
            The input DataFrame containing the features.

        OUTPUT
        ------
        pd.DataFrame
            The preprocessed DataFrame with additional features.
        """
        Wind_Direction_radians = np.radians(df["Wind_Direction_deg"])
        Wind_X = np.sin(Wind_Direction_radians)
        Wind_Y = np.cos(Wind_Direction_radians)

        m100_Elevation_Wind_X = df["100m_Ground_Elevation"] * df["Avg_Wind_Speed"] * Wind_X
        m150_Elevation_Wind_Y = df["150m_Ground_Elevation"] * df["Avg_Wind_Speed"] * Wind_Y
        m150_Humidity_NDVI = df["Relative_Humidity"] * df["150m_NDVI"]
        m150_Traffic_NDBI = df["Traffic_Volume"] * df["150m_NDBI"]
        m300_Building_Wind_X = df["300m_Building_Height"] * df["Avg_Wind_Speed"] * Wind_X
        m300_Building_Wind_Y = df["300m_Building_Height"] * df["Avg_Wind_Speed"] * Wind_Y
        m300_Elevation_Wind_Y = df["300m_Ground_Elevation"] * df["Avg_Wind_Speed"] * Wind_Y
        m300_BldgHeight_Count = df["300m_Building_Height"] * df["300m_Building_Count"]
        m300_TotalBuildingArea_NDVI = df["300m_Total_Building_Area_m2"] * df["300m_NDVI"]
        m300_Traffic_NDVI = df["Traffic_Volume"] * df["300m_NDVI"]
        m300_Traffic_NDBI = df["Traffic_Volume"] * df["300m_NDBI"]
        m300_Building_Aspect_Ratio = df["300m_Building_Height"] / np.sqrt(df["300m_Total_Building_Area_m2"] + 1e-6)
        m300_Sky_View_Factor = 1 - df["300m_Building_Density"]
        m300_Canopy_Cover_Ratio = df["300m_NDVI"] / (df["300m_Building_Density"] + 1e-6)
        m300_GHG_Proxy = df["300m_Building_Count"] * df["Traffic_Volume"] * df["Solar_Flux"] 

        output = {
            "50m_1NPCRI": df["150m_NPCRI"],
            "100m_Elevation_Wind_X": m100_Elevation_Wind_X,
            "150m_Traffic_Volume": df["Traffic_Volume"],
            "150m_Elevation_Wind_Y": m150_Elevation_Wind_Y,
            "150m_Humidity_NDVI": m150_Humidity_NDVI,
            "150m_Traffic_NDBI": m150_Traffic_NDBI,
            "300m_SI": df["300m_SI"],
            "300m_NPCRI": df["300m_NPCRI"],
            "300m_Coastal_Aerosol": df["300m_Coastal_Aerosol"],
            "300m_Total_Building_Area_m2": df["300m_Total_Building_Area_m2"],
            "300m_Building_Construction_Year": df["300m_Building_Construction_Year"],
            "300m_Ground_Elevation": df["300m_Ground_Elevation"],
            "300m_Building_Wind_X": m300_Building_Wind_X,
            "300m_Building_Wind_Y": m300_Building_Wind_Y,
            "300m_Elevation_Wind_Y": m300_Elevation_Wind_Y,
            "300m_BldgHeight_Count": m300_BldgHeight_Count,
            "300m_TotalBuildingArea_NDVI": m300_TotalBuildingArea_NDVI,
            "300m_Traffic_NDVI": m300_Traffic_NDVI,
            "300m_Traffic_NDBI": m300_Traffic_NDBI,
            "300m_Building_Aspect_Ratio": m300_Building_Aspect_Ratio,
            "300m_Sky_View_Factor": m300_Sky_View_Factor,
            "300m_Canopy_Cover_Ratio": m300_Canopy_Cover_Ratio,
            "300m_GHG_Proxy": m300_GHG_Proxy
        }

        output = pd.DataFrame(output, index=[0])

        return output

    def scale(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply the scaler used to train the model to the new data.

        INPUT
        -----
        X: pd.DataFrame - The data to be scaled.

        OUTPUT
        ------
        np.ndarray - The scaled data.
        """
        return self.scaler.transform(X)

    def compute_shap_values(self, X):
        """
        Computes SHAP values for the record.
        """
        # Compute SHAP values
        shap_values = self.explainer.shap_values(X, check_additivity=False) if self.explainer_type == shap.DeepExplainer else self.explainer.shap_values(X)
        
        # Apply squeeze only if the array has three dimensions and the last dimension is 1
        if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
            shap_values = np.squeeze(shap_values)

        return shap_values

    def predict(self, X: pd.DataFrame, location=(None, None)) -> dict:
        """
        Make a prediction on one sample and explain the prediction using SHAP.

        INPUT
        -----
        X: pd.DataFrame - The data to predict a UHI index for (must be one sample).
        location: tuple (longitude, latitude) - Optional location data.

        OUTPUT
        ------
        dict - A dictionary containing the predicted UHI index and SHAP reasoning.
        """
        if X.shape[0] != 1:
            raise ValueError(f"Input array must contain only one sample, but {X.shape[0]} samples were found.")

        # Preprocess and scale input data
        X_processed = self.preprocess(X)
        X_scaled = self.scale(X_processed).reshape(1, -1)

        # Predict UHI index
        y_pred = self.model.predict(X_scaled)
        uhi = y_pred[0][0] if y_pred.ndim == 2 else y_pred[0]

        # Compute SHAP values
        shap_values = self.compute_shap_values(X_scaled)

        # Extract expected base value, Ensure expected_value is a single value (not tensor)
        if self.explainer_type == shap.DeepExplainer:
            expected_value = np.array(self.explainer.expected_value)
        else:
            expected_value = self.explainer.expected_value

        # Extract single value if expected_value is an array
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[0]

        # Compute SHAP-based final prediction
        shap_final_prediction = expected_value + sum(shap_values)

        # Structure feature contributions
        feature_contributions = [
            {
                "feature": feature,
                "shap_value": value,
                "impact": "increase" if value > 0 else "decrease"
            }
            for feature, value in zip(self.feature_names, shap_values)
        ]

        # Create the final output
        prediction_output = {
            "longitude": location[0],
            "latitude": location[1],
            "predicted_uhi_index": uhi,
            "base_value": expected_value,
            "shap_final_prediction": shap_final_prediction,
            "uhi_status": "Urban Heat Island" if shap_final_prediction > 1 else "Cooler Region",
            "feature_contributions": feature_contributions,
        }

        return prediction_output
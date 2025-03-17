import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

class UhiModel:
    """
    Urban Heat Island Model Class that can predict new instances

    INPUTS
    ---
    model_path: the path to the model file
    scaler_path: the path to the standard scaler file
    """
    def __init__(self, model_path, scaler_path):
        self.model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
    
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
        Wind_X = np.sin(df["Wind_Direction"])
        Wind_Y = np.cos(df["Wind_Direction"])

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
            "50m_1NPCRI": df["50m_1NPCRI"],
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
        
    def scale(self, X):
        """
        Apply the scaler used to train the model to the new data

        INPUT
        -----
        X: the data to be scaled
        
        OUTPUT
        ------
        returns the scaled data
        """

        new_data_scaled = self.scaler.transform(X)

        return new_data_scaled

    def predict(self, X: pd.DataFrame) -> float:
        """
        Make a prediction on one sample using the loaded model.

        INPUT
        -----
        X: pd.DataFrame
            The data to predict a UHI index for. Must contain only one sample.

        OUTPUT
        ------
        str:
            Predicted UHI index.
        """

        # Check that input contains only one sample
        if X.shape[0] != 1:
            raise ValueError(f"Input array must contain only one sample, but {X.shape[0]} samples were found")
        
        # Preprocess the input data to create new features
        X_processed = self.preprocess(X)

        # Scale the input data
        X_scaled = self.scale(X_processed)

        # Make prediction
        y_pred = self.model.predict(X_scaled)

        # Extract the predicted UHI index (assuming it's a single value)
        uhi = y_pred[0][0] if y_pred.ndim == 2 else y_pred[0]

        # Return UHI
        return uhi
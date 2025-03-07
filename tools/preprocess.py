"""This module contains tools for preprocessing data."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import skew
from scipy.stats import boxcox
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from tools.environment import BUFFER_DISTANCES, TARGET_VARIABLE

def load_and_prepare_data(data, scaler=None, split=False, test_size=0.3, random_state=42):
    """
    Load and prepare dataset for analysis, optionally scaling features and splitting the dataset
    while preserving DataFrame format.

    Parameters:
        data (str or pd.DataFrame): Path to the CSV file or a Pandas DataFrame.
        scaler (object, optional): A scaler instance (e.g., StandardScaler) with a transform or fit_transform method.
                                    If provided, the scaler will be applied to the features.
        split (bool, optional): If True, split the data into training and validation sets.
        test_size (float, optional): Proportion of the dataset to include in the validation split (default 0.2).
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        If split is False:
            - X (pd.DataFrame): Features, optionally scaled.
            - y (pd.Series): Target variable.
        If split is True:
            - X_train (pd.DataFrame), X_valid (pd.DataFrame): Training and validation features, optionally scaled.
            - y_train (pd.Series), y_valid (pd.Series): Training and validation target variables.
    """
    # Load data if a file path is provided
    if isinstance(data, str):
        print(f"Loading data from {data}")
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        print("Using provided DataFrame.")
        df = data.copy()
    else:
        raise ValueError("Invalid input: `data` must be a file path (str) or a Pandas DataFrame.")
    
    # Separate features and target
    y = df['UHI']
    X = df.drop('UHI', axis=1)
    
    # Remove constant columns
    X = X.loc[:, X.std() != 0]
    
    # Scale features if a scaler is provided
    if scaler is not None:
        try:
            scaled_values = scaler.transform(X)
        except Exception:
            scaled_values = scaler.fit_transform(X)
        # Convert back to a DataFrame with original columns and index
        X = pd.DataFrame(scaled_values, columns=X.columns, index=X.index)
    
    if split:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_valid, y_train, y_valid
    else:
        return X, y

def categorize_UHI(df):
    """
    Convert UHI (continuous variable) to a categorical variable based on its distribution.
    Values less than (mean - std) are categorized as 'cooler',
    values between (mean - std) and (mean + std) are 'same_as_mean',
    and values greater than (mean + std) are 'hotter'.

    A UHI index value of 1.0 suggests the local temperature is the same as the mean temperature 
    of all collected data points. UHI index values above 1.0 are consistent with hotspots above 
    mean temperature values and UHI index values below 1.0 are consistent with cooler locations 
    in the city.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the UHI column.

    Returns:
        new_df (pd.DataFrame): DataFrame with the UHI column converted to a categorical variable.
    """
    # Make a copy of the DataFrame.
    new_df = df.copy()
    
    # Calculate the mean and standard deviation of UHI.
    mean_uhi = new_df['UHI'].mean()
    std_uhi = new_df['UHI'].std()
    
    # Define conditions based on one standard deviation from the mean.
    conditions = [
        (new_df['UHI'] < mean_uhi - std_uhi),
        ((new_df['UHI'] >= mean_uhi - std_uhi) & (new_df['UHI'] <= mean_uhi + std_uhi)),
        (new_df['UHI'] > mean_uhi + std_uhi)
    ]
    
    # Define the corresponding categories.
    choices = ['cooler', 'same_as_mean', 'hotter']
    
    # Use np.select to create a new categorical column.
    new_df['UHI_category'] = np.select(conditions, choices, default='same_as_mean')
    
    return new_df

def preprocess_data(df):
  """
  Preprocess the input buffer dataset.

  Parameters:
      df (pd.DataFrame): The input buffer dataset.

  Returns:
      pd.DataFrame: The preprocessed dataset.
  """
  # Convert wind direction from degrees to radians
  df["Wind Direction [radians]"] = np.radians(df["Wind Direction [degrees]"])

  # Compute sine and cosine for directional components
  df["Wind_X"] = np.sin(df["Wind Direction [radians]"])  # East-West wind component
  df["Wind_Y"] = np.cos(df["Wind Direction [radians]"])  # North-South wind component

  # Interaction: Building Height * Wind Speed * Wind Direction
  df["Building_Wind_X"] = df["Building_Height"] * df["Avg Wind Speed [m/s]"] * df["Wind_X"]
  df["Building_Wind_Y"] = df["Building_Height"] * df["Avg Wind Speed [m/s]"] * df["Wind_Y"]

  # Interaction: Elevation * Wind Speed * Wind Direction
  df["Elevation_Wind_X"] = df["Ground_Elevation"] * df["Avg Wind Speed [m/s]"] * df["Wind_X"]
  df["Elevation_Wind_Y"] = df["Ground_Elevation"] * df["Avg Wind Speed [m/s]"] * df["Wind_Y"]

  # Interaction: Building count * height
  df["BldgHeight_Count"] = df["Building_Height"] * df["Building_Count"]
    
  # Interaction: Urbanization vs Vegetation
  df["BuildingDensity_NDVI"] = df["Building_Density"] * df["NDVI"]
  df["TotalBuildingArea_NDVI"] = df["Total_Building_Area_m2"] * df["NDVI"]
  df["Traffic_NDVI"] = df["Traffic_Volume"] * df["NDVI"]

  # Interaction: Climate interactions w/ urbanization
  df["Temp_BuildingDensity"] = df["Air Temp at Surface [degC]"] * df["Building_Density"]

  # Interaction: Humidity vs Vegetation
  df["Humidity_NDVI"] = df["Relative Humidity [percent]"] * df["NDVI"]
  df["Humidity_NDMI"] = df["Relative Humidity [percent]"] * df["NDMI"]

  # Interaction: Traffic & Built Environment
  df["Traffic_NDBI"] = df["Traffic_Volume"] * df["NDBI"]
  df["Traffic_BuildingDensity"] = df["Traffic_Volume"] * df["Building_Density"]

  # Interaction: Age of Buildings
  df["BuildingAge_Temp"] = df["Building_Construction_Year"] * df["Air Temp at Surface [degC]"]

  # Building & Urban Form Features
  df["Building_Aspect_Ratio"] = df["Building_Height"] / np.sqrt(df["Total_Building_Area_m2"] + 1e-6)  # Avoid division by zero
  df["Sky_View_Factor"] = 1 - df["Building_Density"]
  df["Urban_Material_Index"] = df["Building_Construction_Year"] * df["NDBI"]
  df["Permeable_Impermeable_Ratio"] = df["NDWI"] / (df["NDBI"] + df["SI"] + 1e-6)

  # Vegetation & Surface Features
  df["Land_Surface_Albedo"] = 1 - df["SI"]
  df["Canopy_Cover_Ratio"] = df["NDVI"] / (df["Building_Density"] + 1e-6)
  df["Soil_Moisture_Index"] = df["NDMI"] / (1 + df["NDWI"] + 1e-6)
  
  # Human Activity & Emissions
  df["Traffic_Heat_Emission"] = df["Traffic_Volume"] * df["Air Temp at Surface [degC]"]
  df["GHG_Proxy"] = df["Building_Count"] * df["Traffic_Volume"] * df["Solar Flux [W/m^2]"]
    
  return df

def apply_winsorization(df, limits=(0.01, 0.01)):
    """
    Apply Winsorization to all numerical columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): The dataset containing numerical features.
        limits (tuple): The lower and upper quantile limits for Winsorization.

    Returns:
        pd.DataFrame: The Winsorized dataset.
    """
    df_winsorized = df.copy()  # Create a copy to avoid modifying original data
    
    # Apply Winsorization to all numeric columns
    for col in df_winsorized.select_dtypes(include=['number']).columns:
        df_winsorized[col] = winsorize(df_winsorized[col], limits=limits)

    return df_winsorized

def apply_boxcox_transformation(df, threshold, exclude_cols=[]):
    """
    Apply Box-Cox transformation to specified DataFrame based on skew threshold.

    Parameters:
        df (pd.DataFrame): The dataset containing numerical features.
        threshold (float): The absolute value of the skew threshold.
        exclude_cols (list): List of column names to exclude from transformation.

    Returns:
        pd.DataFrame: The dataset with Box-Cox transformed columns.
    """
    df_transformed = df.copy()  # Create a copy to avoid modifying original data

    # get cols to apply transformation
    boxcox_cols = [col for col in df_transformed.columns if abs(skew(df_transformed[col])) > threshold]
    if exclude_cols:
        boxcox_cols = [col for col in boxcox_cols if col not in exclude_cols]
    
    for col in boxcox_cols:
        # Ensure all values are positive before applying Box-Cox
        if (df_transformed[col] <= 0).any():
            df_transformed[col] += abs(df_transformed[col].min()) + 1  # Shift to positive

        # Apply Box-Cox transformation
        df_transformed[col], _ = boxcox(df_transformed[col])

    return df_transformed

def load_and_preprocess_data(data, scaler=None, split=False, test_size=0.3, random_state=42):
    """
    Load and preprocess dataset for analysis, optionally scaling features and splitting the dataset
    while preserving DataFrame format.

    Parameters:
        data (str or pd.DataFrame): Path to the CSV file or a Pandas DataFrame.
        scaler (object, optional): A scaler instance (e.g., StandardScaler) with a transform or fit_transform method.
                                    If provided, the scaler will be applied to the features. If not, a new StandardScaler will be used.
        split (bool, optional): If True, split the data into training and validation sets.
        test_size (float, optional): Proportion of the dataset to include in the validation split (default 0.2).
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        If split is False:
            - X (pd.DataFrame): Features, optionally scaled.
            - y (pd.Series): Target variable.
        If split is True:
            - X_train (pd.DataFrame), X_valid (pd.DataFrame): Training and validation features, optionally scaled.
            - y_train (pd.Series), y_valid (pd.Series): Training and validation target variables.
        - scaler (object): The fitted scaler instance.
    """
    # Load data if a file path is provided
    if isinstance(data, str):
        print(f"Loading data from {data}")
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        print("Using provided DataFrame.")
        df = data.copy()
    else:
        raise ValueError("Invalid input: `data` must be a file path (str) or a Pandas DataFrame.")

    #preprocess data
    df = preprocess_data(df)
    
    # Separate features and target
    y = df['UHI']
    X = df.drop('UHI', axis=1)
    
    # Remove constant columns
    X = X.loc[:, X.std() != 0]
    
    # Scale features if a scaler is provided
    if scaler is not None:
        try:
            scaled_values = scaler.transform(X)
        except Exception:
            scaled_values = scaler.fit_transform(X)
        # Convert back to a DataFrame with original columns and index
        X = pd.DataFrame(scaled_values, columns=X.columns, index=X.index)
    else:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    if split:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_valid, y_train, y_valid, scaler
    else:
        return X, y, scaler
    
def apply_preprocessing_mixed_buffers(train_combined, test_combined, buffer_distances=BUFFER_DISTANCES):
    """
    Apply `preprocess_data()` to each buffer's subset of variables in both train and test datasets.
    The generated features are prefixed with the corresponding buffer distance.

    Parameters:
        train_combined (pd.DataFrame): The merged training dataset with prefixed buffer features.
        test_combined (pd.DataFrame): The merged testing dataset with prefixed buffer features.

    Returns:
        pd.DataFrame, pd.DataFrame: Updated train and test datasets with prefixed engineered features.
    """
    # Extract shared columns that should remain unchanged
    train_shared_columns = [TARGET_VARIABLE]
    test_shared_columns = ['Longitude', 'Latitude']

    # Initialize lists to store processed buffer data
    processed_train_parts = [train_combined[train_shared_columns].copy()]
    processed_test_parts = [test_combined[test_shared_columns].copy()]

    for buffer_dist in buffer_distances:
        # Extract only this buffer's features
        buffer_train = train_combined.filter(like=f"{buffer_dist}m_").copy()
        buffer_test = test_combined.filter(like=f"{buffer_dist}m_").copy()

        if buffer_train.empty or buffer_test.empty:
            print(f"Skipping {buffer_dist}m buffer as no matching columns found.")
            continue

        # Remove buffer prefixes temporarily for preprocessing
        buffer_train.columns = [col.replace(f"{buffer_dist}m_", "") for col in buffer_train.columns]
        buffer_test.columns = [col.replace(f"{buffer_dist}m_", "") for col in buffer_test.columns]

        # Apply preprocessing function
        buffer_train = preprocess_data(buffer_train)
        buffer_test = preprocess_data(buffer_test)

        # Reapply buffer prefixes to new engineered features
        buffer_train = buffer_train.add_prefix(f"{buffer_dist}m_")
        buffer_test = buffer_test.add_prefix(f"{buffer_dist}m_")

        # Append processed parts
        processed_train_parts.append(buffer_train)
        processed_test_parts.append(buffer_test)

    # Concatenate all processed parts along columns
    final_train = pd.concat(processed_train_parts, axis=1)
    final_test = pd.concat(processed_test_parts, axis=1)

    print(f"\nFinal Processed Training Dataset Shape: {final_train.shape}")
    print(f"Final Processed Testing Dataset Shape: {final_test.shape}")

    return final_train, final_test
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare_data(filepath, scaler=None, split=False, test_size=0.3, random_state=42):
    """
    Load and prepare dataset for analysis, optionally scaling features and splitting the dataset
    while preserving DataFrame format.

    Parameters:
        filepath (str): Path to the CSV file.
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
    print(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
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
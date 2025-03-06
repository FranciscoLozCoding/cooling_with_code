import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =====================================
# Configuration
# =====================================
TARGET_VARIABLE = 'Air Temp at Surface [degC]'  # Target variable to predict
BUFFER_DISTANCES = [50, 100, 150]  # Exclude 300m buffer
RANDOM_STATE = 42  # For reproducibility
TEST_SIZE = 0.3  # Proportion of data for testing

# =====================================
# Data Loading
# =====================================
def load_data():
    """Load buffer datasets for specified buffer distances."""
    buffer_datasets = {}
    
    # Try to load each buffer dataset
    for buffer_dist in BUFFER_DISTANCES:
        file_path = f'{buffer_dist}m_buffer_dataset.csv'
        test_path = f'{buffer_dist}m_buffer_test_dataset.csv'
        
        # Check for training data
        if os.path.exists(file_path):
            print(f"Loading {file_path}...")
            buffer_datasets[buffer_dist] = pd.read_csv(file_path)
        # Fall back to test data if training not available
        elif os.path.exists(test_path):
            print(f"Training data not found. Using test data from {test_path}...")
            buffer_datasets[buffer_dist] = pd.read_csv(test_path)
    
    # If no datasets found, raise an error
    if not buffer_datasets:
        raise FileNotFoundError("No buffer datasets found. Please check file paths.")
    
    print(f"Loaded data for {len(buffer_datasets)} buffer distances: {list(buffer_datasets.keys())}")
    return buffer_datasets

def check_target_variable(buffer_datasets, target_variable):
    """Check if the target variable exists in all datasets."""
    valid_buffers = []
    
    for buffer_dist, data in buffer_datasets.items():
        if target_variable in data.columns:
            print(f"Target variable '{target_variable}' found in {buffer_dist}m buffer dataset.")
            valid_buffers.append(buffer_dist)
        else:
            print(f"Warning: Target variable '{target_variable}' not found in {buffer_dist}m buffer dataset.")
            print(f"Columns in {buffer_dist}m buffer dataset:")
            for col in data.columns:
                print(f"  {col}")
            
            # Try to find similar columns
            similar_cols = [col for col in data.columns if 'temp' in col.lower() or '[degc]' in col.lower()]
            if similar_cols:
                print(f"Similar columns found: {similar_cols}")
    
    return valid_buffers

def filter_datasets(buffer_datasets, valid_buffers):
    """Keep only datasets that contain the target variable."""
    return {buffer_dist: data for buffer_dist, data in buffer_datasets.items() if buffer_dist in valid_buffers}

# =====================================
# Land Use Regression Model
# =====================================
def find_best_buffer_per_variable(buffer_datasets, target_variable):
    """Find the best buffer distance for each variable."""
    # Identify all variables across all datasets
    all_variables = set()
    for data in buffer_datasets.values():
        for col in data.columns:
            if col != target_variable and col.lower() not in ['longitude', 'latitude', 'id']:
                all_variables.add(col)
    
    print(f"Found {len(all_variables)} unique variables across all buffer distances.")
    
    # Dictionary to track best buffer for each variable
    best_buffer_per_variable = {}
    variable_buffer_scores = {}
    
    # Evaluate each variable across available buffers
    for variable in sorted(all_variables):
        print(f"Evaluating {variable}...")
        variable_buffer_scores[variable] = {}
        
        for buffer_dist, data in buffer_datasets.items():
            # Skip if variable not in this dataset
            if variable not in data.columns:
                continue
            
            # Skip if variable has missing values
            if data[variable].isnull().any():
                continue
            
            # Create a simple model to test this variable
            X = data[[variable]]
            y = data[target_variable]
            
            # Scale the variable
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Use Ridge regression
            model = Ridge(alpha=1.0)
            
            try:
                # Cross-validate
                scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
                avg_score = np.mean(scores)
                
                # Store the score
                variable_buffer_scores[variable][buffer_dist] = avg_score
                
                print(f"  {buffer_dist}m buffer: R² = {avg_score:.4f}")
            except Exception as e:
                print(f"  Error evaluating {variable} at {buffer_dist}m: {e}")
        
        # Find best buffer for this variable
        if variable_buffer_scores[variable]:
            best_buffer = max(
                variable_buffer_scores[variable],
                key=variable_buffer_scores[variable].get
            )
            best_score = variable_buffer_scores[variable][best_buffer]
            
            # Only include variables with meaningful predictive power
            if best_score > 0.1:
                best_buffer_per_variable[variable] = best_buffer
                print(f"  Best buffer for {variable}: {best_buffer}m (R² = {best_score:.4f})")
            else:
                print(f"  {variable} had no substantial R² values. Skipping.")
    
    return best_buffer_per_variable, variable_buffer_scores

def build_combined_dataset(buffer_datasets, target_variable, best_buffer_per_variable):
    """Build a combined dataset using the best buffer for each variable."""
    if not best_buffer_per_variable:
        print("No suitable variables found for model building.")
        return None
        
    # Start with an empty dataframe
    combined_data = None
    
    # For each variable, get data from the best buffer
    for variable, buffer_dist in best_buffer_per_variable.items():
        data = buffer_datasets[buffer_dist]
        
        # For the first variable, initialize with ID columns and target
        if combined_data is None:
            # Identify ID columns (usually Longitude and Latitude)
            id_columns = [col for col in data.columns 
                         if col.lower() in ['longitude', 'latitude', 'id']]
            
            # Create the initial dataframe with ID columns and target
            combined_data = data[id_columns + [target_variable]].copy()
        
        # Add the variable with its buffer distance as suffix
        combined_data[f"{variable}_{buffer_dist}m"] = data[variable]
    
    return combined_data

def train_final_model(combined_data, target_variable):
    """Train the final regression model using the combined dataset."""
    # Get predictor columns (those with buffer suffix)
    predictor_cols = [col for col in combined_data.columns 
                     if '_' in col and col != target_variable]
    
    print(f"Training model with {len(predictor_cols)} predictors")
    
    # Prepare data
    X = combined_data[predictor_cols]
    y = combined_data[target_variable]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Ridge regression with cross-validation for alpha
    alphas = np.logspace(-3, 3, 20)
    model = RidgeCV(alphas=alphas, cv=5)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Get selected alpha
    print(f"Selected alpha for Ridge regression: {model.alpha_}")
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Evaluate model
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Print results
    print(f"\n=== Model Performance ===")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Feature coefficients
    feature_coeffs = pd.DataFrame({
        'feature': predictor_cols,
        'coefficient': model.coef_
    })
    feature_coeffs['abs_coefficient'] = np.abs(feature_coeffs['coefficient'])
    feature_coeffs = feature_coeffs.sort_values('abs_coefficient', ascending=False)
    
    print("\nTop 10 most important variables:")
    for i, row in feature_coeffs.head(10).iterrows():
        print(f"  {row['feature']}: {row['coefficient']:.6f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'rmse': rmse,
        'feature_coeffs': feature_coeffs
    }

# =====================================
# Visualization
# =====================================
def visualize_buffer_selection(best_buffer_per_variable):
    """Visualize which buffer was selected for each variable."""
    if not best_buffer_per_variable:
        return
    
    # Create a DataFrame for the buffer selection results
    buffer_df = pd.DataFrame({
        'Variable': list(best_buffer_per_variable.keys()),
        'Best_Buffer_m': list(best_buffer_per_variable.values())
    })
    
    # Count variables per buffer
    buffer_counts = buffer_df['Best_Buffer_m'].value_counts().sort_index()
    
    # Plot variable count by buffer
    plt.figure(figsize=(10, 6))
    buffer_counts.plot(kind='bar')
    plt.title('Number of Variables per Buffer Distance')
    plt.xlabel('Buffer Distance (m)')
    plt.ylabel('Number of Variables')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('variables_per_buffer.png', dpi=300)
    plt.close()
    
    # Plot each variable and its best buffer
    plt.figure(figsize=(12, max(8, len(best_buffer_per_variable) * 0.3)))
    buffer_df = buffer_df.sort_values('Best_Buffer_m')
    sns.barplot(x='Best_Buffer_m', y='Variable', data=buffer_df)
    plt.title('Best Buffer Distance Selected per Variable')
    plt.tight_layout()
    plt.savefig('best_buffer_per_variable.png', dpi=300)
    plt.close()

def visualize_model_results(results):
    """Visualize model performance."""
    if not results:
        return
    
    # Unpack results
    y_train = results['y_train']
    y_test = results['y_test']
    y_train_pred = results['y_train_pred']
    y_test_pred = results['y_test_pred']
    feature_coeffs = results['feature_coeffs']
    
    # Create actual vs predicted plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Training data
    ax1.scatter(y_train, y_train_pred, alpha=0.6)
    min_val = min(y_train.min(), y_train_pred.min())
    max_val = max(y_train.max(), y_train_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title(f'Training Data: R² = {results["train_r2"]:.4f}')
    
    # Testing data
    ax2.scatter(y_test, y_test_pred, alpha=0.6)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title(f'Testing Data: R² = {results["test_r2"]:.4f}')
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300)
    plt.close()
    
    # Feature importance plot
    plt.figure(figsize=(12, 8))
    top_features = feature_coeffs.head(15)
    
    plt.barh(top_features['feature'], top_features['abs_coefficient'])
    plt.xlabel('Absolute Coefficient')
    plt.ylabel('Feature')
    plt.title('Top 15 Features by Importance')
    
    # Add coefficient values to the bars
    for i, v in enumerate(top_features['abs_coefficient']):
        plt.text(v, i, f'  {top_features["coefficient"].iloc[i]:.4f}', va='center')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.close()

# =====================================
# Main Workflow
# =====================================
def main():
    print("=== Land Use Regression (Excluding 300m Buffer) ===")
    
    # Step 1: Load data
    print("\nStep 1: Loading data...")
    buffer_datasets = load_data()
    
    if not buffer_datasets:
        print("No datasets found. Exiting.")
        return
    
    # Step 2: Check target variable
    print(f"\nStep 2: Checking for target variable '{TARGET_VARIABLE}'...")
    valid_buffers = check_target_variable(buffer_datasets, TARGET_VARIABLE)
    
    if not valid_buffers:
        print("Target variable not found in any dataset. Exiting.")
        return
    
    # Filter datasets to include only those with the target variable
    buffer_datasets = filter_datasets(buffer_datasets, valid_buffers)
    
    # Step 3: Find best buffer per variable
    print("\nStep 3: Finding optimal buffer distance for each variable...")
    best_buffer_per_variable, variable_buffer_scores = find_best_buffer_per_variable(
        buffer_datasets, TARGET_VARIABLE
    )
    
    # Step 4: Visualize buffer selection
    print("\nStep 4: Visualizing buffer selection...")
    visualize_buffer_selection(best_buffer_per_variable)
    
    # Step 5: Build combined dataset
    print("\nStep 5: Building combined dataset with optimal buffers...")
    combined_data = build_combined_dataset(
        buffer_datasets, TARGET_VARIABLE, best_buffer_per_variable
    )
    
    if combined_data is None:
        print("Could not build combined dataset. Exiting.")
        return
    
    print(f"Combined dataset shape: {combined_data.shape}")
    combined_data.to_csv('combined_dataset_no_300m.csv', index=False)
    
    # Step 6: Train final model
    print("\nStep 6: Training final model...")
    results = train_final_model(combined_data, TARGET_VARIABLE)
    
    # Step 7: Visualize model results
    print("\nStep 7: Visualizing model results...")
    visualize_model_results(results)
    
    # Save results
    results['feature_coeffs'].to_csv('feature_coefficients.csv', index=False)
    
    print("\nAnalysis complete!")
    print("Results saved to CSV files and visualizations saved as PNG files.")

if __name__ == "__main__":
    main()
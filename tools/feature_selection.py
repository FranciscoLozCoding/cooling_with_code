"""
Feature importance analysis across different buffer zones using actual UHI data
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import tools.preprocess

def load_and_prepare_data(filepath, scaler=None, split=False, test_size=0.3, random_state=42):
    """
    This function has been deprecated. Please use the load_and_prepare_data function from the preprocess module instead.
    """
    return tools.preprocess.load_and_prepare_data(filepath, scaler, split, test_size, random_state)

def get_feature_importance(X, y, n_estimators=100, random_state=42):
    """Calculate feature importance using Random Forest with UHI data"""
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Fit model with actual UHI values
    rf_model.fit(X, y)
    
    # Calculate feature importance and standard deviation
    importances = rf_model.feature_importances_
    importances_std = np.std([
        tree.feature_importances_ for tree in rf_model.estimators_
    ], axis=0)
    
    return pd.DataFrame({
        'feature': X.columns,
        'importance': importances,
        'std': importances_std
    }).sort_values('importance', ascending=True)

def plot_feature_importance_comparison(importances_50m, importances_100m, importances_150m):
    """Create comparison plot of feature importance across buffer zones"""
    # Set up figure with adjusted size for all features
    plt.figure(figsize=(15, max(12, len(importances_50m) * 0.4)))
    
    # Combine importance data
    combined_importance = pd.DataFrame({
        'feature': importances_50m['feature'],
        '50m': importances_50m['importance'],
        '100m': importances_100m['importance'],
        '150m': importances_150m['importance'],
        '50m_std': importances_50m['std'],
        '100m_std': importances_100m['std'],
        '150m_std': importances_150m['std']
    })
    
    # Calculate mean importance and sort
    combined_importance['mean_importance'] = combined_importance[['50m', '100m', '150m']].mean(axis=1)
    combined_importance = combined_importance.sort_values('mean_importance', ascending=True)
    
    # Set up bar positions
    y_pos = np.arange(len(combined_importance))
    width = 0.25
    
    # Create bars with error bars
    plt.barh(y_pos - width, combined_importance['50m'], width, 
             xerr=combined_importance['50m_std'],
             label='50m', alpha=0.8, color='#2ecc71')
    plt.barh(y_pos, combined_importance['100m'], width,
             xerr=combined_importance['100m_std'],
             label='100m', alpha=0.8, color='#3498db')
    plt.barh(y_pos + width, combined_importance['150m'], width,
             xerr=combined_importance['150m_std'],
             label='150m', alpha=0.8, color='#e74c3c')
    
    # Customize plot
    plt.xlabel('Feature Importance (with standard deviation)', fontsize=12)
    plt.title('Feature Importance for UHI Prediction Across Buffer Zones', pad=20, fontsize=14)
    plt.yticks(y_pos, combined_importance['feature'], fontsize=10)
    
    # Add grid
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    
    # Add legend
    plt.legend(title='Buffer Zone', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    return plt

def perm_importance(rf_model, X_valid, y_valid, n_repeats=5, random_state=42, scoring='r2'):
    """
    Calculate permutation importance for a fitted Random Forest model and return a DataFrame 
    with feature names, mean importance, and standard deviation.
    
    Parameters:
        rf_model: Fitted Random Forest model.
        X_valid: Validation set features (expected as a DataFrame for feature names).
        y_valid: Validation set target.
        n_repeats (int): Number of times to shuffle each feature.
        random_state (int): Random seed for reproducibility.
        scoring (str): Scoring metric to use for permutation importance.
        
    Returns:
        DataFrame: Contains columns 'feature', 'importance', and 'std', sorted in ascending order by importance.
    """
    # Compute permutation importance on the validation set.
    perm_imp = permutation_importance(
        rf_model, 
        X_valid, 
        y_valid, 
        n_repeats=n_repeats,          
        random_state=random_state,
        scoring=scoring
    )
    
    # Extract importance and standard deviation.
    importances = perm_imp.importances_mean
    std = perm_imp.importances_std
    
    # Get feature names; if X_valid is a DataFrame, use its columns.
    if hasattr(X_valid, 'columns'):
        feature_names = list(X_valid.columns)
    else:
        feature_names = [f'Feature {i}' for i in range(X_valid.shape[1])]
    
    # Create a DataFrame for permutation importance.
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'std': std
    })
    
    # Sort the DataFrame by importance (ascending order) for easier plotting.
    perm_df.sort_values('importance', ascending=True, inplace=True)
    
    return perm_df

def plot_perm_importance_comparison(perm_importance_50m, perm_importance_100m, perm_importance_150m):
    """
    Create a comparison plot of permutation feature importances across buffer zones.

    Parameters:
        perm_importance_50m (DataFrame): Contains 'feature', 'importance', and 'std' for the 50m buffer.
        perm_importance_100m (DataFrame): Contains 'feature', 'importance', and 'std' for the 100m buffer.
        perm_importance_150m (DataFrame): Contains 'feature', 'importance', and 'std' for the 150m buffer.

    Returns:
        matplotlib.pyplot: The plot object.
    """

    # Combine the permutation importance data from each buffer zone into one DataFrame.
    combined_importance = pd.DataFrame({
        'feature': perm_importance_50m['feature'],
        '50m': perm_importance_50m['importance'],
        '100m': perm_importance_100m['importance'],
        '150m': perm_importance_150m['importance'],
        '50m_std': perm_importance_50m['std'],
        '100m_std': perm_importance_100m['std'],
        '150m_std': perm_importance_150m['std']
    })

    # Calculate the mean importance across the three buffer zones and sort the DataFrame.
    combined_importance['mean_importance'] = combined_importance[['50m', '100m', '150m']].mean(axis=1)
    combined_importance = combined_importance.sort_values('mean_importance', ascending=True)

    # Set up bar positions.
    y_pos = np.arange(len(combined_importance))
    width = 0.25

    # Create the plot.
    plt.figure(figsize=(15, max(12, len(combined_importance) * 0.4)))

    # Create horizontal bars for each buffer zone with error bars.
    plt.barh(y_pos - width, combined_importance['50m'], width, 
             xerr=combined_importance['50m_std'],
             label='50m', alpha=0.8, color='#2ecc71')
    plt.barh(y_pos, combined_importance['100m'], width,
             xerr=combined_importance['100m_std'],
             label='100m', alpha=0.8, color='#3498db')
    plt.barh(y_pos + width, combined_importance['150m'], width,
             xerr=combined_importance['150m_std'],
             label='150m', alpha=0.8, color='#e74c3c')

    # Customize the plot.
    plt.xlabel('Permutation Feature Importance (with std. deviation)', fontsize=12)
    plt.title('Permutation Feature Importance for UHI Prediction Across Buffer Zones', pad=20, fontsize=14)
    plt.yticks(y_pos, combined_importance['feature'], fontsize=10)
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.legend(title='Buffer Zone', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    return plt

def lasso_importance(X_train, y_train, alphas, cv_folds=5, random_state=42):
    """
    Calculate Lasso regression feature importance using GridSearchCV with KFold cross-validation.
    Make sure to scale the features before using this function.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        alphas (array-like): List or array of alpha values to test.
        cv_folds (int, optional): Number of folds for cross-validation (default is 5).
        random_state (int, optional): Random state for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with columns 'feature', 'coefficient', and 'abs_coef'
                      sorted in ascending order by 'abs_coef'.
    """
    # Set up the parameter grid for alpha.
    params = {"alpha": alphas}
    
    # Set up KFold cross-validation.
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Initialize Lasso with the given random state.
    lasso = Lasso(random_state=random_state)
    
    # Perform grid search to find the best alpha.
    grid = GridSearchCV(lasso, param_grid=params, cv=kf)
    grid.fit(X_train, y_train)
    
    # Get the best estimator from grid search.
    best_lasso = grid.best_estimator_
    
    # Extract coefficients.
    coefs = best_lasso.coef_
    
    # Build a DataFrame for feature importance.
    lasso_df = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': coefs,
        'abs_coef': np.abs(coefs)
    })
    
    # Sort features by absolute coefficient value.
    lasso_df.sort_values('abs_coef', ascending=True, inplace=True)
    return lasso_df

def plot_lasso_importance(lasso_df, title='Lasso Regression Feature Importance'):
    """
    Plot Lasso regression feature importance as a horizontal bar chart.
    
    Parameters:
        lasso_df (pd.DataFrame): DataFrame returned by lasso_importance containing
                                 'feature', 'coefficient', and 'abs_coef'.
    
    Returns:
        matplotlib.pyplot: The plot object.
    """
    # Determine bar positions
    y_pos = np.arange(len(lasso_df))
    
    # Create horizontal bar plot using absolute coefficient values.
    plt.figure(figsize=(10, 6))
    plt.barh(y_pos, lasso_df['abs_coef'], align='center', color='teal')
    plt.yticks(y_pos, lasso_df['feature'], fontsize=10)
    plt.xlabel('Absolute Coefficient')
    plt.title(title)
    plt.tight_layout()
    return plt

def rfe_importance(X, y, estimator, n_features_to_select=1, step=1):
    """
    Calculate feature ranking using Recursive Feature Elimination (RFE).

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        estimator: A fitted estimator (e.g., RandomForestRegressor, Lasso, etc.) used by RFE.
        n_features_to_select (int, optional): Number of features to select; defaults to 1.
        step (int, optional): Number of features to remove at each iteration; defaults to 1.

    Returns:
        pd.DataFrame: DataFrame with columns 'feature' and 'ranking', where a lower ranking indicates greater importance.
    """
    # Initialize and fit RFE on the provided data.
    rfe = RFE(estimator, n_features_to_select=n_features_to_select, step=step)
    rfe.fit(X, y)
    
    # rfe.ranking_ gives the rank of each feature (1 is the best)
    ranking = rfe.ranking_
    
    # Create a DataFrame with the feature names and their corresponding rankings.
    rfe_df = pd.DataFrame({
        'feature': X.columns,
        'ranking': ranking
    })
    
    # Sort the DataFrame so that the best features (lowest ranking) come first.
    rfe_df.sort_values('ranking', ascending=True, inplace=True)
    return rfe_df

def plot_rfe_importance(rfe_df, title='RFE Feature Importance'):
    """
    Plot the RFE feature ranking as a horizontal bar chart.
    
    Parameters:
        rfe_df (pd.DataFrame): DataFrame from rfe_importance with columns 'feature' and 'ranking'.
    
    Returns:
        matplotlib.pyplot: The plot object.
    """
    # Sort the DataFrame so that the best features (rank=1) appear at the top.
    rfe_df = rfe_df.sort_values('ranking', ascending=True)
    y_pos = np.arange(len(rfe_df))
    
    plt.figure(figsize=(10, 6))
    plt.barh(y_pos, rfe_df['ranking'], align='center', color='blue')
    plt.yticks(y_pos, rfe_df['feature'])
    plt.xlabel('RFE Ranking (1 = Best)')
    plt.title(title)
    plt.gca().invert_yaxis()  # Best features on top
    plt.tight_layout()
    return plt

def main():
    try:
        # Load datasets with actual UHI values
        print("Loading datasets...")
        X_50m, y_50m = load_and_prepare_data('50m_buffer_dataset.csv')
        X_100m, y_100m = load_and_prepare_data('100m_buffer_dataset.csv')
        X_150m, y_150m = load_and_prepare_data('150m_buffer_dataset.csv')
        
        # Calculate importance using actual UHI values
        print("Calculating feature importance...")
        importance_50m = get_feature_importance(X_50m, y_50m)
        importance_100m = get_feature_importance(X_100m, y_100m)
        importance_150m = get_feature_importance(X_150m, y_150m)
        
        # Create visualization
        print("Creating visualization...")
        plt = plot_feature_importance_comparison(importance_50m, importance_100m, importance_150m)
        
        # Save plot
        print("Saving visualization...")
        plt.savefig('uhi_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print all features with their importance scores
        print("\nComplete Feature Importance Rankings (averaged across buffer zones):")
        combined_importance = pd.DataFrame({
            'feature': importance_50m['feature'],
            '50m': importance_50m['importance'],
            '100m': importance_100m['importance'],
            '150m': importance_150m['importance']
        })
        combined_importance['mean_importance'] = combined_importance[['50m', '100m', '150m']].mean(axis=1)
        all_features = combined_importance.sort_values('mean_importance', ascending=False)
        
        # Format and save detailed results to CSV
        all_features.to_csv('uhi_feature_importance_rankings.csv', index=False)
        
        # Print formatted output with more detail
        pd.set_option('display.float_format', '{:.6f}'.format)
        print(all_features.to_string(index=False))
        print("\nVisualization saved as 'uhi_feature_importance_comparison.png'")
        print("Detailed rankings saved as 'uhi_feature_importance_rankings.csv'")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find one or more dataset files. Please ensure all CSV files are in the correct directory.")
        print(f"Details: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
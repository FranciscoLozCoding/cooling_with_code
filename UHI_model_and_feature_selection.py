"""
Feature importance analysis across different buffer zones using actual UHI data
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def load_and_prepare_data(filepath):
    """Load and prepare dataset for analysis"""
    print(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Separate features and target
    y = df['UHI']
    X = df.drop('UHI', axis=1)
    
    # Remove constant columns
    X = X.loc[:, X.std() != 0]
    
    return X, y

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
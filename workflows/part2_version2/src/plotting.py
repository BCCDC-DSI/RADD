import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import os
import shap

def plot_metrics(df, model_col, metric_col, dataset_col, output_dir, filename='metrics.png'):
    """
    Plot the Metrics on the Training and Test data as a bar plot
    
    Parameters
    ----------
        df : DataFrame
            A DataFrame of metrics (MSE + MAPE) with Train and Test identifiers
        model_col : String
            The column with the model names
        metric_col : String
            The Metric column for the y axis
        dataset_col : String
            The Dataset column for hues
        output_dir : String
            Output directory to write the plot to
        filename : String
            The File to save as .png
    Returns
    -------
        <None>
    """
    plt.figure(figsize=(10,10))
    sns.barplot(data=df , x = model_col, y = metric_col, hue=dataset_col)
    plt.xticks(rotation = 45)
    plt.title('Barplot of ' + metric_col + ' on Train and Test Data')
    plt.savefig(os.path.join(output_dir, filename))
    return None

def plot_shap(shap_values, X_test, features_final, name, output_dir, filename):
    # Create the modified plots
    plot_size = (10, 8)
    plt.figure(figsize = plot_size)
    shap.summary_plot(shap_values, features=X_test, show = False, feature_names = features_final)
    plt.title('SHAP summary plot of ' + name)
    plt.savefig(os.path.join(output_dir, 'SHAP_summary_' +  str(name) + '_' + filename + '.png'), bbox_inches='tight')

def plot_error_bins(df_long, output_dir, filename='error_plots.png'):
    # Plotting with seaborn
    error_bin_order = ['<= 1.0', '> 1.0 and <= 2.0', '> 2.0']
    
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df_long, x='model', hue='error_bin', hue_order=error_bin_order)
    plt.title('Error Distribution Across Models')
    plt.xlabel('Model')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Error Bin')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
SEED=42
import sys
import os
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the src directory to the Python path
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

# Path to the config file
config_path = os.path.join(current_dir, 'config', 'config.yaml')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.dates as mdates
import shap  # package used to calculate Shap values
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.metrics import mean_absolute_percentage_error
import train_test
import random
import utils
import plotting
from sklearn import linear_model
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import catboost
from catboost import CatBoostRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso
import tensorflow as tf
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import yaml

tf.compat.v1.disable_v2_behavior()

logger = logging.getLogger('RADD')

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

## Load configurations
model_index = config['model_index']
model_y = config['model_y']
model_X = config['model_X']

# Directories
working_dir = config['working_dir']
raw_data_dir = config['raw_data_dir']
linking_dir = config['linking_dir']
output_dir = config['output_dir']
default_output_dir = config['default_output_dir']

# Filenames
raw_data_filename = config['raw_data_filename']
linking_filename = config['linking_filename']
smiles_dict_filename = config['smiles_dict_filename']
output_filename = config['output_filename']

# ML Pipeline
model_names = config['model_names']

def summary_stats_models(models, train_df, test_df, X_train, y_train, X_test, y_test, output_dir):
    """
    Save the summary stats of all the LHA models in CSV format

    Parameters
    ----------
        models : dict{name}{model}
            Dictionary of model name and ML model
        X_train : numpy
            Training Data
        y_train : numpy
            Training Predictions
        X_test : numpy
            Test Data
        y_test : numpy
            Test predictions
        output_dir : String
            Path to output folder
    Returns
    -------
        <None>
    """
    train_mse_list = []
    test_mse_list = []
    train_rsquare_list = []
    test_rsquare_list = []
    train_mape_list = []
    test_mape_list = []
    
    best_hyper_params = []
    for _, model in models.items():

        # MSE
        train_mse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
        test_mse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

        # R square
        train_rsquare = r2_score(y_train, model.predict(X_train))
        test_rsquare = r2_score(y_test, model.predict(X_test))

        # MAPE
        train_mape = mean_absolute_percentage_error(y_train, model.predict(X_train))
        test_mape = mean_absolute_percentage_error(y_test, model.predict(X_test))

        # Add it to lists
        train_mse_list.append(train_mse)
        test_mse_list.append(test_mse)
        
        train_rsquare_list.append(train_rsquare)
        test_rsquare_list.append(test_rsquare)

        train_mape_list.append(train_mape)
        test_mape_list.append(test_mape)

        # Also store the Best Hyperparams for each model
        # Check the model type and append hyperparameters accordingly
        if isinstance(model, (GridSearchCV, RandomizedSearchCV)):
        # For models trained with GridSearchCV or RandomizedSearchCV
            best_hyper_params.append(model.best_params_)
        elif isinstance(model, KerasRegressor):
            # You may need to define a method to extract hyperparameters from KerasRegressor
            keras_hyper_params = model.get_params()
            best_hyper_params.append(keras_hyper_params)
        elif isinstance(model, Sequential):
            # For Sequential models, handle the absence of best_params_
            best_hyper_params.append("N/A - Pure Keras Model")
        else:
            # Handle other types of models or append a placeholder
            best_hyper_params.append("N/A - Unsupported Model Type")
        #logger.info(str(models[i].best_params_))

    hyperparams_df = pd.DataFrame(data=[best_hyper_params], columns = model_names)
    hyperparams_df.to_csv(os.path.join(output_dir, 'hyperparams.csv'), index=False)


    # Convert to Long form
    train_mse_df = utils.create_long_metrics_df(model_names, train_mse_list, 'RMSE', 'Train')
    test_mse_df = utils.create_long_metrics_df(model_names, test_mse_list, 'RMSE', 'Test')

    train_rsquare_df = utils.create_long_metrics_df(model_names, train_rsquare_list, 'R2', 'Train')
    test_rsquare_df = utils.create_long_metrics_df(model_names, test_rsquare_list, 'R2', 'Test')

    train_mape_df = utils.create_long_metrics_df(model_names, train_mape_list, 'MAPE', 'Train')
    test_mape_df = utils.create_long_metrics_df(model_names, test_mape_list, 'MAPE', 'Test')
    
    # Output Plots for Manuscript
    mse_df = pd.concat([train_mse_df, test_mse_df])
    rsquare_df = pd.concat([train_rsquare_df, test_rsquare_df])
    mape_df = pd.concat([train_mape_df, test_mape_df])

    plotting.plot_metrics(mse_df, 'Model', 'RMSE', 'Dataset', output_dir, 'rmse_metrics.png')
    plotting.plot_metrics(rsquare_df, 'Model', 'R2', 'Dataset', output_dir, 'r2_metrics.png')
    plotting.plot_metrics(mape_df, 'Model', 'MAPE', 'Dataset', output_dir, 'mape_metrics.png')

    # Write out the MSE in a file format
    summary_df = pd.concat([mse_df, rsquare_df, mape_df], axis=1)
    
    # Remove Duplicate columns
    summary_df = summary_df.loc[:,~summary_df.columns.duplicated()].copy()
    summary_df.to_csv(os.path.join(output_dir, 'summary_stats.csv'), index=False)

def shap_summary_models(ml_dict, features, features_processed, X_test, y_test, output_dir, filename='test'):
    """
    Plot the SHAP Summary plots of all ML models and save them to a file
    Save the converted census descriptions with their relative Denominators
    in a sorted fashion.
    
    Parameters
    ----------
        ml_dict : Dict{String}{pickle.model}
            A Dictionary of ML models with their names and the models themselves
        features : List[String]
            A list of the ML features
        features_processed : List[String]
            A list of the preprocessed ML features
        X_test : np.array
            NumPy array of Test Features
        y_test : np.array
            NumPy array of test predictors
        output_dir : String
            Path to write output file to
    Returns
    -------
        <None>
    """
    for name, model in ml_dict.items():
        
        try:
            best_model = model.best_estimator_
        except:
            best_model = model
        if isinstance(best_model, catboost.CatBoostRegressor):
            explainer = shap.Explainer(best_model)
        elif isinstance(best_model, (tf.keras.Model, KerasRegressor)):  
            # This will handle both tf.keras.Model and KerasRegressor
            keras_model = best_model.model if isinstance(best_model, KerasRegressor) else best_model
            explainer = shap.DeepExplainer(keras_model, X_test)
        elif isinstance(best_model, (LinearRegression, Lasso)):
            explainer = shap.LinearExplainer(best_model, X_test)
        else:
            explainer = shap.TreeExplainer(best_model)

        shap_values = explainer.shap_values(X_test)
        shap_values = np.array(shap_values)
        # Neural Net check
        if shap_values.ndim > 2:
            shap_values = np.squeeze(shap_values)
        
        features_extra = []
        for string in features_processed:
            features_extra.append(string.replace('num__', ''))
        model_result = pd.DataFrame(shap_values, columns = features_extra)
        vals = np.abs(model_result).mean(0)

        # Modifying how you create feature_importance to ensure correct feature matching
        feature_importance = vals.reset_index()
        feature_importance.columns = ['Feature Name', 'feature_importance_vals']
        feature_importance['model'] = name
        features_final = features_processed

        # Create the modified plots
        plotting.plot_shap(shap_values, X_test, features_final, name, output_dir, filename)
        
    return None
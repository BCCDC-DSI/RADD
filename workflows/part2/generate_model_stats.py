# Generate the model stats like error bins, visuals and create the prediction data
import sys
import os
import yaml
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the src directory to the Python path
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

# Path to the config file
config_path = os.path.join(current_dir, 'config', 'config.yaml')

import train
import test
import utils
import warnings
import plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
import logging
import pickle
from sklearn import preprocessing
import math
import time
# Import the ML Models
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost
from tensorflow.keras.models import Sequential, model_from_json
import tensorflow as tf

with open('config/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def create_output_dirs(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

## Load configurations
model_index = config['model_index']
model_y = config['model_y']
test_model_index = config['test_model_index']
test_model_y = config['test_model_y']
model_X = config['model_X']
model_names = config['model_names']

def create_norm_X(df, vectorizer, processor):
    """
    Parameters
    ----------
        df : pd.DataFrame
            DataFrame of SMILES data to encode. Note: Must have a SMILES column.
        vectorizer : SMILESVectorizer()
            Smiles2Vec vectorizer, must be instantiated before it is used.
        processor : pickle.preprocessor
            Preprocessor used during ML pipeline.
    Returns
    -------
        norm_X : Normalized X Train values for prediction.
    """
    # Create the encoded dataframe
    df_vectorized, _ = vectorizer.transform(df[model_X].to_list())

    # Flatten and create feature names
    df_flattened = train.flatten_and_create_feature_names(df_vectorized)

    # Combine with additional features (assuming no additional features in this example)
    combined_df = train.combine_with_additional_features(df_flattened, None)

    norm_X = processor.transform(combined_df)

    return norm_X

def make_error_bins(df, model_index, model_y):
    """
    Make Error bins for the supplied dataframe of retention times
    
    Parameters
    ----------
        df : pd.DataFrame
            DataFrame of SMILES data, Retention Times and predicted retention times
        model_index : String
            The name of the Model index column
        model_y : String
            The name of the prediction (usually Retention Time in mins.)
    Returns
    -------
        df_long : pd.DataFrame
            DataFrame in Long format which has the compound, actual, predicted retention times, with over/under prediction, errors and error bins.
    """
    # Calculate errors
    for model in model_names:
        df[f'{model}_error'] = abs(df[model+'_prediction'] - df[model_y])

    # Convert to long format manually
    long_data = []
    for index, row in df.iterrows():
        for model in model_names:
            if model != 'model_y':
                long_data.append({
                    'Compound': row[model_index],
                    'SMILES' : row[model_X],
                    'Actual RT': row[model_y],
                    'model': model,
                    'prediction': row[model+'_prediction'],
                    'error': row[f'{model}_error'],
                    'under_over': 'Under' if row[model+'_prediction'] < row[model_y] else 'Over'
                })

    df_long = pd.DataFrame(long_data)

    # Function to bin errors
    def bin_errors(error):
        if error <= 1.0:
            return '<= 1.0'
        elif 1.0 < error <= 2.0:
            return '> 1.0 and <= 2.0'
        else:
            return '> 2.0'

    # Apply binning
    df_long['error_bin'] = df_long['error'].apply(bin_errors)
    return df_long

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', default='output', help='Output dir for results')
    parser.add_argument('-d', '--database', default='', help='Path to the database the model was trained on.')
    parser.add_argument('-t', '--test_database', default='', help='Path to the database you wish to deploy the model on.')
    parser.add_argument('-m', '--load_models', default=None, help='Use loaded models and skip rerunning the pipeline')
    parser.add_argument('-p', '--load_preprocessor', default=None, help='Path to the preprocessor file')
    return parser.parse_args()

def main():
    args = _parse_args()
    create_output_dirs(args.output_dir)
    database = pd.read_csv(args.database)
    test_database = pd.read_csv(args.test_database)

    # ensure the dataframes are clean
    database.dropna(subset=[model_X], inplace=True)
    test_database.dropna(subset=[model_X], inplace=True)

    # check the database sizes
    if database.shape[0] < test_database.shape[0]:
        print(f'The test database has {test_database.shape[0]} compounds, note that you are using a model trained on a lower number of compounds.')

    # Load the ML Models
    all_models = []
    for root, _, files in os.walk(os.path.join(args.load_models)):
        for file in files:
            if file.endswith('.json'):
                    json_file = open(root + '/' + file, 'r')
                    loaded_model_json = json_file.read()
                    loaded_model = model_from_json(loaded_model_json)
                    all_models.append(loaded_model)
                    json_file.close() 
            elif file.endswith('.pkl'):
                with open(root + '/' + file, 'rb') as f:
                    all_models.append(pickle.load(f))
            elif file.endswith('.h5'):
                neural_net_weights = os.path.join(root, file)

    for i in all_models:
        if isinstance(i, tf.keras.Model):
            i.load_weights(neural_net_weights)

    # Load the preprocessor
    with open(args.load_preprocessor, 'rb') as f:
        processor = pickle.load(f)

    # Instaniate the Encoder
    vectorizer = train.SMILESVectorizer()

    # This Line is key, we always fit to the databases SMILES
    vectorizer.fit(database[model_X].to_list())

    # Create norm_Xs
    database_norm_X = create_norm_X(database, vectorizer, processor)
    test_database_norm_X = create_norm_X(test_database, vectorizer, processor)

    prediction_col_names = []
    # Add the predictions to the original dataframe
    for i,name in enumerate(model_names):
        prediction_col_names.append(name + '_prediction')
        database[name + '_prediction'] = all_models[i].predict(database_norm_X)
        test_database[name + '_prediction'] = all_models[i].predict(test_database_norm_X)
    
    # Create the long dataframes
    database_plot = make_error_bins(database, model_index, model_y)
    test_database_plot = make_error_bins(test_database, test_model_index, test_model_y)

    # Generate error bin plots
    plotting.plot_error_bins(database_plot, args.output_dir, 'error_db.png')
    plotting.plot_error_bins(test_database_plot, args.output_dir, 'error_test_db.png')

    # Output the long dataframes to the output folder
    database_plot.to_csv(os.path.join(args.output_dir, 'db_long.csv'), index=False)
    test_database_plot.to_csv(os.path.join(args.output_dir, 'test_db_long.csv'), index=False)

if __name__ == '__main__':
    main()
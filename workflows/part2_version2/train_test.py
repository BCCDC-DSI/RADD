#!/usr/bin/env python
# Train and test
import sys
import os
import yaml
sys.path.append('src/')
import train
import test
import utils
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
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

def config_logger(output_dir):
    logger = logging.getLogger("RADD")
    logger.setLevel(logging.DEBUG)
    # create handlers
    fh = logging.FileHandler(os.path.join(output_dir, 'train_test_log.txt'))
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', default='output', help='Output dir for results')
    parser.add_argument('-d', '--data', default='', help='Path to CSV file of data, see README for format.')
    parser.add_argument('-t', '--train_size', default=0.8, help='Control the training data size, see README for format.')
    parser.add_argument('-x', '--load_models', default=None, help='Use loaded models and skip rerunning the pipeline')
    return parser.parse_args()

def create_output_dirs(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def format_data(df):
    """
    Format the Data into X, y to setup different variables for the Machine Learning pipeline

    Parameters
    ----------
    df : DataFrame
        DataFrame to separate into X,y
    
    Returns
    -------
    (X, y) : tuple(numpy, numpy)
        NumPy arrays of features and predictor
        
    """
    y = df[model_y]
    y = y.to_numpy()
    X = df[[model_X]]
    return X,y

def main():
    args = _parse_args()
    create_output_dirs(args.output_dir)
    logger = config_logger(args.output_dir)

    # preprocess and split into train and test
    logger.info("Loading the data...")

    # Load the data
    all_data = pd.read_csv(args.data, index_col=False)
    split = math.floor(float(args.train_size)*len(all_data)) # default 80% train
    logger.info(f'Splitting the data into a {float(args.train_size)*100}:{100-float(args.train_size)*100} split')

    # Train and Test data
    train_data = all_data[:split]
    test_data  = all_data[split:]

    # Keep the Test and Train DataFrames (for plots)
    train_data_raw = train_data.copy()
    test_data_raw = test_data.copy()
    
    # print some stats on data
    logger.info("Total samples: {}".format(len(all_data)))
    logger.info("    Train samples: {}".format(len(train_data)))
    logger.info("    Test samples: {}".format(len(test_data)))

    # Split into X_train and y_train
    X_train, y_train = format_data(train_data)
    X_test, y_test = format_data(test_data)

    # Prepare columns
    features = X_train.columns.to_list()

    # Vectorize SMILES data in the DataFrame
    X_train_vectorized, _ = train.vectorize_smiles(X_train)
    X_test_vectorized, _ = train.vectorize_smiles(X_test)
    
    # Flatten and create feature names
    df_train_flattened = train.flatten_and_create_feature_names(X_train_vectorized)
    df_test_flattened = train.flatten_and_create_feature_names(X_test_vectorized)
    
    # Combine with additional features (assuming no additional features in this example)
    combined_df_train = train.combine_with_additional_features(df_train_flattened, None)
    combined_df_test = train.combine_with_additional_features(df_test_flattened, None)
    
    # Preprocess the training data
    norm_X_train, processor = train.preprocess_combined_df(combined_df_train)
    # Transform the test data using the fitted processor
    norm_X_test = processor.transform(combined_df_test)

    # Retain for SHAPs
    features_processed = processor.get_feature_names_out()

    # Train all models
     # Train all models
    all_models = []
    if args.load_models:
        for root, dirs, files in os.walk(args.load_models):
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
        # Ensure correct ordering
        if not all_models:
            logger.info("The chosen directory does not have models, using latest trained models. See config file.")
            for root, dirs, files in os.walk(os.path.join(default_output_dir, 'models')):
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

    # Load the Neural Net weights in
        for i in all_models:
            if isinstance(i, tf.keras.Model):
                i.load_weights(neural_net_weights)

    else:
        catboost, lasso, lgbm, neural_net,  rf, xgboost = train.train_all_models(norm_X_train, y_train, args.output_dir)
        all_models = [catboost, lasso, lgbm, neural_net, rf, xgboost]

    ml_dict =  dict(zip(model_names, all_models))
    test.summary_stats_models(ml_dict, train_data_raw, test_data_raw, norm_X_train, y_train, norm_X_test, y_test, args.output_dir)
    test.shap_summary_models(ml_dict, features, features_processed, norm_X_test, y_test, args.output_dir)

if __name__ == "__main__":
    main()


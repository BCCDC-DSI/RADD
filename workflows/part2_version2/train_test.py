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

with open('config/ml_config.yaml') as f:
    ml_config = yaml.load(f, Loader=yaml.FullLoader)

global predictor_name
predictor_name = config['predictor_name']
date_name = config['date_name']
LHA_ID = config['LHA_ID']
pipeline_results_dir = config['pipeline_results_dir']

model_names = ml_config['model_names']

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
    y = df[predictor_name]
    y = y.to_numpy()
    col_names = df.columns.tolist()
    col_names.remove(predictor_name)
    X = df[col_names]

    return X,y

def main():
    args = _parse_args()
    create_output_dirs(args.output_dir)
    logger = config_logger(args.output_dir)

    # preprocess and split into train and test
    logger.info("Loading the data...")

    # Load the data (We ignore the date columns - Assume ORDERED)
    all_data = pd.read_csv(args.data, index_col=False)

    # Create the day features
    logger.info("Making day of the week and holiday features")
    all_data = utils.create_day_features(all_data, date_name)

    logger.info("Missingness Analysis...")
    missingness_df = utils.show_missing(all_data)
    logger.info(missingness_df.head())
    if missingness_df[missingness_df['missing'] > 0] is not None:
        logger.info("There is some missingess, Imputation will be implemented at preprocessing")
        warnings.warn('Missing values detected, consider using a complete dataset otherwise Imputation will take place')
    # train test split (Time series version)
    split = math.floor(float(args.train_size)*len(all_data)) # default 80% train
    logger.info("Splitting the data into a " + str(float(args.train_size)*100) + ":" + str(1 - float(args.train_size)*100) + " split")
    train_data = all_data[:split]
    test_data  = all_data[split:]
    rolling_vec = utils.rolling_avg_regressor(all_data)
    rolling_y_train = rolling_vec[:split]
    rolling_y_test = rolling_vec[split:]

    # Keep the Test and Train DataFrames (for plots)
    train_data_raw = train_data.copy()
    test_data_raw = test_data.copy()

    # Retain Dates and LHA IDs as separate variables for plots
    train_dates = train_data[date_name]
    test_dates = test_data[date_name]
    train_dates = pd.to_datetime(train_dates)
    test_dates = pd.to_datetime(test_dates)
    train_ids = train_data[LHA_ID]
    test_ids = test_data[LHA_ID]
    
    # Remove them from the training and test data
    train_data.drop(columns=[date_name, LHA_ID], inplace=True)
    test_data.drop(columns=[date_name, LHA_ID], inplace=True)
    
    # print some stats on data
    logger.info("Total samples: {}".format(len(all_data)))
    logger.info("    Train samples: {}".format(len(train_data)))
    logger.info("    Test samples: {}".format(len(test_data)))

    # Split into X_train and y_train
    X_train, y_train = format_data(train_data)
    X_test, y_test = format_data(test_data)

    # Prepare columns
    features = X_train.columns.to_list()
    df_numerical_features = X_train.select_dtypes(include='number')
    df_categorical_features = X_train.select_dtypes(include='object')

    # preprocess
    features = X_train.columns.to_list()
    numerical_features = df_numerical_features.columns.to_list()
    categorical_features = df_categorical_features.columns.to_list()
    processor = train.fit_processor(X_train[features],numerical_features, categorical_features, args.output_dir)
    norm_X_train = processor.transform(X_train[features])
    norm_X_test = processor.transform(X_test[features])

    # Prediction Preprocess
    y_train = np.array(y_train).reshape(-1,1)
    y_test = np.array(y_test).reshape(-1,1)
    prediction_preprocessor = train.scale_prediction(y_train, args.output_dir)
    norm_y_train = prediction_preprocessor.transform(y_train)
    norm_y_test = prediction_preprocessor.transform(y_test)

    # Retain for SHAPs
    features_processed = processor.get_feature_names_out()

    # Train all models
    # UNRESHAPE THE DATA SO IT WORKS
    norm_y_train = norm_y_train.ravel()
    norm_y_test = norm_y_test.ravel()
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
            for root, dirs, files in os.walk(os.path.join(pipeline_results_dir, default_output_dir, 'models')):
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
        catboost, lasso, lgbm, neural_net,  rf, xgboost = train.train_all_models(norm_X_train, norm_y_train, args.output_dir)
        all_models = [catboost, lasso, lgbm, neural_net, rf, xgboost]

    ml_dict =  dict(zip(model_names, all_models))

    # Load Census Info for testing models
    proportion_dict, description_dict = census.load_census_info()
    # Test all Models
    test.summary_stats_models(ml_dict, train_data_raw, test_data_raw, norm_X_train, norm_y_train, norm_X_test, norm_y_test, rolling_y_test, args.output_dir)
    #test.shap_summary_models(ml_dict, features, features_processed, norm_X_train, norm_y_train, proportion_dict, description_dict, args.output_dir, 'train')
    test.shap_summary_models(ml_dict, features, features_processed, norm_X_test, norm_y_test, proportion_dict, description_dict, args.output_dir)
    

if __name__ == "__main__":
    main()

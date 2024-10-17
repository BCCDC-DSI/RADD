#!/usr/bin/env python
# Train and test
import sys
import os
sys.path.append('src/')
import train
import test
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
from imblearn.over_sampling import RandomOverSampler
# Import the ML Models
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost

global model_names
model_names = ['Lasso', 'Random Forest', 'LGBM', 'CatBoost', 'XGBoost']

def config_logger(output_dir):
    logger = logging.getLogger("COVID Reinfection Pipeline")
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
    parser.add_argument('-t', '--train_size', default=0.7, help='Control the training data size, see README for format.')
    parser.add_argument('-r', '--oversample', default=None, help='Set an oversampling strategy to balance the dataset.' )
    return parser.parse_args()

def create_output_dirs(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def format_data(df):
    """
    Formats the total visits data to X, y
    """
    y = df['episode_reinfection']
    y = y.to_numpy()
    col_names = df.columns.tolist()
    col_names.remove('episode_reinfection')
    X = df[col_names]

    return X,y

def main():
    args = _parse_args()
    create_output_dirs(args.output_dir)
    logger = config_logger(args.output_dir)
    # preprocess and split into train and test
    logger.info("Loading and preprocessing data...")

    # Load the data (We ignore the date columns - Assume ORDERED)
    try:
        all_data = pd.read_csv(args.data, index_col=False, delimiter=',', parse_dates=["collection_date"], encoding='utf-8')
    except:
        all_data = pd.read_excel(args.data, index_col=False, parse_dates=["collection_date"])
    all_data.sort_values(by='collection_date', inplace=True)
    # train test split (Time series version)
    split = math.floor(float(args.train_size)*len(all_data)) # 50% train
    train_data = all_data[:split]
    test_data  = all_data[split:]
    train_dates = train_data['collection_date']
    test_dates = test_data['collection_date']
    train_dates = pd.to_datetime(train_dates)
    test_dates = pd.to_datetime(test_dates)

    train_data.drop(columns=['collection_date'], inplace=True)
    test_data.drop(columns=['collection_date'], inplace=True)
    # print some stats on data
    logger.info("Total samples: {}".format(len(all_data)))
    logger.info("    Train samples: {}".format(len(train_data)))
    logger.info("    Test samples: {}".format(len(test_data)))
    
    # Split into X_train and y_train
    X_train, y_train = format_data(train_data)
    X_test, y_test = format_data(test_data)

    # Apply oversampling (if specified)
    if args.oversample:
        logger.info("Pipeline oversampling enabled")
        try:
            oversampling_strategy = float(args.oversample)
        except:
            logger.info("Non-numeric oversampling provided, defaulting to minority class sampling")
            oversampling_strategy = "minority"
        oversample=RandomOverSampler(sampling_strategy=oversampling_strategy)
        X_train, y_train = oversample.fit_resample(X_train, y_train)
        X_test, y_test = oversample.fit_resample(X_test, y_test)

    df_numerical_features = X_train.select_dtypes(include='number')
    df_categorical_features = X_train.select_dtypes(include='object')

    # preprocess
    features = X_train.columns.to_list()
    numerical_features = df_numerical_features.columns.to_list()
    categorical_features = df_categorical_features.columns.to_list()
    processor = train.fit_processor(X_train[features],numerical_features, categorical_features, args.output_dir)
    norm_X_train = processor.transform(X_train[features])
    norm_X_test = processor.transform(X_test[features])
    features_processed = processor.get_feature_names_out()

    # Train all models
    catboost, lgbm, rf, xgboost = train.train_all_models(norm_X_train, y_train, args.output_dir)

    # Predict
    all_models = [catboost, lgbm, rf, xgboost]
    ml_dict =  dict(zip(model_names, all_models))
    
    # Print and save plots
    test.plot_roc(ml_dict, norm_X_train, y_train, args.output_dir, 'train-roc.png')
    test.plot_roc(ml_dict, norm_X_test, y_test, args.output_dir)
    test.shap_summary_models(ml_dict, features_processed, norm_X_test, args.output_dir)
    test.summary_stats_models(ml_dict, norm_X_test, y_test, args.output_dir)

    
if __name__ == "__main__":
    main()

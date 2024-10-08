import sys
import os
sys.path.append('src/')
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
import train_test
import random
import delong
import pickle

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default='', help='Path to CSV file of data, see README for format.')
    parser.add_argument('-t', '--train_size', default=0.7, help='Control the training data size, see README for format.')
    parser.add_argument('-r', '--oversample', default=None, help='Set an oversampling strategy to balance the dataset.' )
    return parser.parse_args()

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
    all_data = pd.read_csv(args.data, index_col=False, parse_dates=["collection_date"])

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


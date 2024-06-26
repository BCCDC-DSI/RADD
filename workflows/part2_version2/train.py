SEED = 2022
#!/usr/bin/env python
# Train and test
import sys
import os
sys.path.append('src/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.decomposition import PCA
import logging
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import (
    TimeSeriesSplit,
    cross_val_score,
    cross_validate,
    train_test_split,
)
import time
import create_neural_net
import yaml 
# Import the ML Models
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam, RMSprop
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import model_from_json
import json

logger = logging.getLogger('RADD')

with open('config/ml_config.yml') as f:
    ml_config = yaml.load(f, Loader=yaml.FullLoader)


def scale_prediction(y_train, output_dir):
    """
    Applies a MinMaxScaler to the prediction

    Paramters
    ---------
        y_train : numpy
            Prediction
    
    Returns
    -------
        preprocessor : sklearn.Preprocessor
             sklearn preprocessor fit on the training set
    """
    preprocessor = MinMaxScaler(feature_range=(0,1))
    logger.info("Preprocessing y_train")
    norm_y_train = preprocessor.fit(y_train)
    with open(os.path.join(output_dir, 'prediction_processor.pkl'),'wb') as f:
        pickle.dump(preprocessor, f)
    return preprocessor

def fit_processor(X_train, numeric_features, categorical_features, output_dir):
    """
    Applies Simple Imputer to Categorical Features
    Applies One Hot Encoding to Categorical Features
    Applies Quantile Scaling to Numeric Features
    Returns and writes pickle file of the complete preprocessor

    Parameters
    ----------
    X_train : numpy
        Training Data in NumPy format
    numeric_features : list[string]
        List of Numeric Features
    categorical_features: list[string]
        List of Categorical Features
    output_dir: string
        Output directory to write the preprocessor to
    
    Returns
    -------
    preprocessor : sklearn.Preprocessor
        sklearn preprocessor fit on the training set
    """
    pipe_num = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler',  MinMaxScaler(feature_range=(-1, 1)))
    ])
    pipe_cat = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')),
        ('impute', SimpleImputer(strategy='constant', fill_value=0))
        
    ]) 
    preprocessor = ColumnTransformer([
        ('num', pipe_num, numeric_features),
        ('cat', pipe_cat, categorical_features)
    ])
    logger.info("Preprocessing X_train")
    norm_X_train = preprocessor.fit(X_train)
    with open(os.path.join(output_dir, 'processor.pkl'),'wb') as f:
        pickle.dump(preprocessor, f)
    return preprocessor

def time_series_cross_val(model, X_train, y_train, param_search):
    """
    Performs Time Series Split Cross Validation on Training Data (set to 10 splits)
    
    Parameters
    ----------
    model : sklearn.Model
        Machine Learning model (Linear or Tree based)
    X_train : numpy
        Training Data
    y_train : numpy
        Training Predictions
    param_search : dict{string: list[string] or string: list[int]}
        Grid Search parameters
    
    Returns
    -------
    best_model : sklearn.Model
        Best model by maximising objective function on validation set
    """

    tscv = TimeSeriesSplit(n_splits=10)
    bscv = blocked_time_series_split.BlockingTimeSeriesSplit(n_splits=10, margins=14)
    gsearch = GridSearchCV(estimator=model, cv=bscv,
                            param_grid=param_search, n_jobs = -1)
    best_model = gsearch.fit(X_train, y_train)

    return best_model

def train_all_models(X_train, y_train, output_dir):
    """
    Train All the ML models on X_train and y_train
    Apply cross_validation in a select rolling fashion

    Parameters
    ----------
    X_train : numpy
        Training Data
    y_train : numpy
        Training Predictions
    output_dir : string
        Output directory to write the models to

    Returns
    -------
    models : tuple[pm.model, sklearn.Model, sklearn.Model, sklearn.Model, keras.model, sklearn.Model, sklearn.Model]
        Returns the ML models as a tuple to use in the test script
    """
    logger.info("Training all models")

    # Initialise models
    logger.info("Writing models to specified output folder")
    if not os.path.exists(os.path.join(output_dir, "models")):
        os.makedirs(os.path.join(output_dir, "models"))

    #  Lasso (set it as a Poisson Regression instead)
    logger.info("Training Lasso Model")
    lasso_params = {
        'alpha' : [0.001, 0.01, 0.1],
        'max_iter': [500, 1000, 2000]
    }
    lasso = time_series_cross_val(linear_model.Lasso(), X_train, y_train, lasso_params)
    with open(os.path.join(output_dir, 'models/lasso_model.pkl'), 'wb') as f:
        pickle.dump(lasso, f)

    # Random Forest
    logger.info("Training Random Forest Regressor")
    rf_params = {
        'max_depth': [2, 4, 8],
        'n_estimators': [4, 16, 64],
        'min_samples_split': [8, 16, 32]
    }
    rf = time_series_cross_val(RandomForestRegressor(), X_train, y_train, rf_params)
    with open(os.path.join(output_dir, 'models/rf_model.pkl'), 'wb') as f:
        pickle.dump(rf, f)

    # LGBM
    logger.info("Training LGBM Regressor")
    lgbm_params = {
        'num_leaves': [4,8,16,32,64,128],
         'max_depth': [2,4,8],
         'min_data_in_leaf': [2,4,8,16,32]
    }
    lgbm = time_series_cross_val(lgb.LGBMRegressor(), X_train, y_train, lgbm_params)
    with open(os.path.join(output_dir, 'models/lgbm_model.pkl'), 'wb') as f:
        pickle.dump(lgbm, f)

    # Neural Network
    logger.info("Training Neural Network")
    # Assuming your data has explicit time steps
    input_dim = X_train.shape[1]  # Replace with the correct dimension for your data
    base_neural_net = KerasRegressor(build_fn=create_neural_net.build_fn(input_dim), epochs=50, batch_size=32, verbose=1)

    neural_net_params = {
        'epochs' : [50, 100, 150],
        'batch_size' : [16, 32, 64]
    }

    neural_net = time_series_cross_val(base_neural_net, X_train, y_train, neural_net_params)
    neural_net_json = neural_net.best_estimator_.model.to_json() 
    with open(os.path.join(output_dir, 'models/neural_net_model.json'), 'w') as f:
        f.write(neural_net_json)

    # Save the model weights to HDF5
    neural_net.best_estimator_.model.save_weights(os.path.join(output_dir,"models/neural_net_model_weights.h5"))

    # Catboost
    logger.info("Training Catboost Regressor")
    catboost_params = {
        'depth':[1,5,10],
        'iterations':[50, 100, 200],
        'learning_rate':[0.001, 0.01, 0.1],
        'l2_leaf_reg':[5, 10, 20],
        'border_count':[25, 50, 75],
        'thread_count':[4]
        }
    catboost = time_series_cross_val(CatBoostRegressor(), X_train, y_train, catboost_params)
    with open(os.path.join(output_dir, 'models/catboost_model.pkl'), 'wb') as f:
        pickle.dump(catboost, f)

    # XGBoost
    logger.info("Training XGBoost Regressor")
    xgboost_params = {
        # Parameters that we are going to tune.
        'max_depth':[6, 10],
        'min_child_weight': [1, 3],
        'eta':[.3, .7],
        'subsample': [1],
        'colsample_bytree': [1],
        'alpha' : [0.01, 0.03],
        'lambda' : [2, 4],
        # Other parameters
        'objective':['reg:squarederror']
    }
    xgboost = time_series_cross_val(XGBRegressor(), X_train, y_train, xgboost_params)
    with open(os.path.join(output_dir, 'models/xgboost_model.pkl'), 'wb') as f:
        pickle.dump(xgboost, f)

    return catboost, lasso, lgbm, neural_net, rf, xgboost
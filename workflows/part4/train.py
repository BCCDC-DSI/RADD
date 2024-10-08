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
from sklearn.model_selection import KFold
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
# Import the ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

logger = logging.getLogger('COVID Reinfection Pipeline')

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
        ('scaler',  QuantileTransformer(output_distribution = 'normal', random_state=SEED))
    ])
    pipe_cat = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False)),
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

def cross_val(model, X_train, y_train, param_search):
    """
    Performs k-fold cross validation
    
    Parameters
    ----------
    model : sklearn.Model
        Machine Learning model (Linear or Tree based)
    X_train : numpy.ndarray
        Training Data
    y_train : numpy.ndarray
        Training Predictions
    param_search : dict
        Grid Search parameters
    
    Returns
    -------
    best_model : sklearn.Model
        Best model by maximizing objective function on validation set
    """
    
    # Define the k-fold cross-validation strategy
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Initialize GridSearchCV with the specified model, cross-validation, and parameter grid
    gsearch = GridSearchCV(estimator=model, cv=kf, param_grid=param_search, n_jobs=-1)
    
    # Fit GridSearchCV to find the best model based on the given parameters
    best_model = gsearch.fit(X_train, y_train)
    
    return best_model

def train_all_models(X_train, y_train, output_dir):
    """
    Train All the ML models on X_train and y_train
    Apply cross_validation in a select rolling fashion
    """
    logger.info("Training all models")
    # Initialise models
    logger.info("Writing models to specified output folder")
    if not os.path.exists(os.path.join(output_dir, "models")):
        os.makedirs(os.path.join(output_dir, "models"))
    model_output_dir = os.path.join(output_dir, 'models')
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    # Random Forest
    logger.info("Training Random Forest Classifier")
    rf_params = {
        'class_weight': ['balanced'],
        'max_depth': [2, 4, 8, 16, 32],
        'n_estimators': [4, 16, 64, 256],
        'min_samples_split': [2, 4, 8, 16, 32]
    }
    rf = cross_val(RandomForestClassifier(), X_train, y_train, rf_params)
    with open(os.path.join(output_dir, 'models/rf_model.pkl'), 'wb') as f:
        pickle.dump(rf, f)

    # LGBM
    logger.info("Training LGBM Classifier")
    lgbm_params = {
        'class_weight': ['balanced'],
        'num_leaves': [4,8,16,32,64,128],
         'max_depth': [2,4,8],
         'min_data_in_leaf': [2,4,8,16,32]
    }
    lgbm = cross_val(lgb.LGBMClassifier(), X_train, y_train, lgbm_params)
    with open(os.path.join(output_dir, 'models/lgbm_model.pkl'), 'wb') as f:
        pickle.dump(lgbm, f)

    # Catboost
    logger.info("Training Catboost Classifier")
    catboost_params = {
            'auto_class_weights': ['Balanced', 'SqrtBalanced'],
            'depth':[1,5,10],
            'iterations':[50, 100, 200],
            'learning_rate':[0.001, 0.01, 0.1],
            'l2_leaf_reg':[1,5, 10],
            'border_count':[10, 50, 100],
            'thread_count':[4]}
    catboost = cross_val(CatBoostClassifier(), X_train, y_train, catboost_params)
    with open(os.path.join(output_dir, 'models/catboost_model.pkl'), 'wb') as f:
        pickle.dump(catboost, f)

    # XGBoost
    logger.info("Training XGBoost Classifier")
    # Compute the scales for the `scale_pos_weight` param  
    pos_weight = (len(y_train)-y_train.sum() - y_train.sum())/len(y_train)
    xgboost_params = {
        'scale_pos_weight':[pos_weight, np.sqrt(pos_weight)],
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
    xgboost = cross_val(XGBClassifier(), X_train, y_train, xgboost_params)
    with open(os.path.join(output_dir, 'models/xgboost_model.pkl'), 'wb') as f:
        pickle.dump(xgboost, f)

    return catboost, lgbm, rf, xgboost

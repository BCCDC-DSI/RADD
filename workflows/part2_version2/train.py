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
from sklearn.model_selection import KFold
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

def create_char_to_int(smiles):
    """Create a character-to-integer mapping based on the input SMILES strings."""
    unique_chars = set(''.join(smiles))
    char_to_int = {char: i for i, char in enumerate(unique_chars)}
    char_to_int['!'] = len(char_to_int)  # Start character
    char_to_int['E'] = len(char_to_int)  # End character
    char_to_int['UNK'] = len(char_to_int)  # Unknown character
    return char_to_int

def vectorize_smiles(smiles):
    """
    Vectorize a list of SMILES strings into one-hot encoded representations.

    This function converts a list of SMILES strings into a three-dimensional
    one-hot encoded array, suitable for input into neural network models. It
    automatically determines the maximum SMILES length in the dataset and adds
    two additional positions for start ('!') and end ('E') characters.

    Parameters
    ----------
    smiles : list of str
        List of SMILES strings to be vectorized.

    Returns
    -------
    tuple of np.ndarray
        A tuple containing two numpy arrays:
        - The first array is the one-hot encoded input sequences, excluding the end character.
        - The second array is the one-hot encoded output sequences, excluding the start character.
        
    Examples
    --------
    >>> smiles = ["CCO", "NCC", "CCCCCCCCCCC"]
    >>> X, Y = vectorize_smiles(smiles)
    >>> X.shape
    (3, 13, 27)
    >>> Y.shape
    (3, 13, 27)
    Note: for this pipeline Y is a return value we do not utilise.
    """
    char_to_int = create_char_to_int(smiles)

    # Determine the maximum SMILES length
    max_smiles_length = max(len(smile) for smile in smiles)
    embed_length = max_smiles_length + 2  # Add 2 for start ('!') and end ('E') characters
    charset_size = len(char_to_int)
    
    def vectorize(smiles):
        one_hot = np.zeros((len(smiles), embed_length, charset_size), dtype=np.int8)
        for i, smile in enumerate(smiles):
            # encode the start character
            one_hot[i, 0, char_to_int["!"]] = 1
            # encode the rest of the characters
            for j, c in enumerate(smile):
                if c in char_to_int:
                    one_hot[i, j + 1, char_to_int[c]] = 1
                else:
                    one_hot[i, j + 1, char_to_int['UNK']] = 1
            # encode end character
            one_hot[i, len(smile) + 1, char_to_int["E"]] = 1
        # return two, one for input and the other for output
        return one_hot[:, 0:-1, :], one_hot[:, 1:, :]

    return vectorize(smiles)

def flatten_and_create_feature_names(X_vectorized):
    """
    Flatten the 3D one-hot encoded array into a 2D array and create feature names.
    
    Parameters
    ----------
    X_vectorized : np.ndarray
        3D array of shape (samples, max_length, charset_size)
    
    Returns
    -------
    pd.DataFrame
        2D DataFrame with flattened features and meaningful names
    """
    num_samples, max_length, charset_size = X_vectorized.shape
    
    # Flatten the 3D array into a 2D array
    X_flattened = X_vectorized.reshape(num_samples, -1)
    
    # Create feature names
    feature_names = [f"pos_{i}_char_{j}" for i in range(max_length) for j in range(charset_size)]
    
    # Create DataFrame with feature names
    df_flattened = pd.DataFrame(X_flattened, columns=feature_names)
    
    return df_flattened

def combine_with_additional_features(df_flattened, additional_features):
    """
    Combine the flattened DataFrame with additional features.
    
    Parameters
    ----------
    df_flattened : pd.DataFrame
        DataFrame with flattened features
    additional_features : pd.DataFrame
        DataFrame with additional numeric or categorical features
    
    Returns
    -------
    pd.DataFrame
        Combined DataFrame
    """
    combined_df = pd.concat([df_flattened, additional_features], axis=1)
    return combined_df

def preprocess_combined_df(combined_df):
    """
    Preprocess the combined DataFrame.
    
    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined DataFrame with flattened SMILES features and additional features    
    Returns
    -------
    np.ndarray
        Preprocessed array
    ColumnTransformer
        Fitted preprocessor
    """
    # Dynamically identify numeric and categorical features
    numeric_features = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = combined_df.select_dtypes(include=['object']).columns.tolist()
    
    pipe_num = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])
    pipe_cat = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', pipe_num, numeric_features),
        ('cat', pipe_cat, categorical_features)
    ], remainder='passthrough')
    
    X_preprocessed = preprocessor.fit_transform(combined_df)
    
    return X_preprocessed, preprocessor

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
    lasso = cross_val(linear_model.Lasso(), X_train, y_train, lasso_params)
    with open(os.path.join(output_dir, 'models/lasso_model.pkl'), 'wb') as f:
        pickle.dump(lasso, f)

    # Random Forest
    logger.info("Training Random Forest Regressor")
    rf_params = {
        'max_depth': [2, 4, 8],
        'n_estimators': [4, 16, 64],
        'min_samples_split': [8, 16, 32]
    }
    rf = cross_val(RandomForestRegressor(), X_train, y_train, rf_params)
    with open(os.path.join(output_dir, 'models/rf_model.pkl'), 'wb') as f:
        pickle.dump(rf, f)

    # LGBM
    logger.info("Training LGBM Regressor")
    lgbm_params = {
        'num_leaves': [4,8,16,32,64,128],
         'max_depth': [2,4,8],
         'min_data_in_leaf': [2,4,8,16,32]
    }
    lgbm = cross_val(lgb.LGBMRegressor(), X_train, y_train, lgbm_params)
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

    neural_net = cross_val(base_neural_net, X_train, y_train, neural_net_params)
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
    catboost = cross_val(CatBoostRegressor(), X_train, y_train, catboost_params)
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
    xgboost = cross_val(XGBRegressor(), X_train, y_train, xgboost_params)
    with open(os.path.join(output_dir, 'models/xgboost_model.pkl'), 'wb') as f:
        pickle.dump(xgboost, f)

    return catboost, lasso, lgbm, neural_net, rf, xgboost
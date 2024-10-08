import sys
import os
import pandas as pd
import numpy as np
import logging
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam, RMSprop
from keras.wrappers.scikit_learn import KerasRegressor

def create_neural_net_model(input_dim, optimizer='adam', units_1=64, units_2=32):
    """
    Parameters
    ----------               
        input_dim : int
            Dimensions of the Training data (number of features)
        optimizer : String
            Neural Network optimizer
        units_1 : int
            Number of units going into the first layer of the neural network
        units_2 : int
            Number of units going into the second layer of the neural network
    
    Returns
    -------
        model : keras.model
            Baseline Keras neural network model
    """
    model = Sequential()
    model.add(Dense(units=units_1, activation='relu', input_dim=input_dim))
    model.add(Dense(units=units_2, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def build_fn(input_dim, optimizer='adam', units_1=64, units_2=32):
    """
    Parameters
    ----------
        input_dim : int
            Dimensions of the Training data (number of features)
        optimizer : String
            Neural Network Optimizer
        units_1 : int
            Number of units going into the first layer of the neural network
        units_2 : int
            Number of units going into the second layer of the neural network
    Returns
    -------
        model : keras.model
            Baseline Keras neural network model

    """
    def create_model():
        return create_neural_net_model(input_dim, optimizer=optimizer, units_1=units_1, units_2=units_2)
    return create_model
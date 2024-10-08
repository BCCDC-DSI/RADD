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

logger = logging.getLogger('COVID Reinfection Pipeline')

global model_names
model_names = ['Lasso', 'Random Forest', 'LGBM', 'CatBoost', 'XGBoost']


def take_rt_stats(rt, stat = 0.5):
    rt_df = rt.groupby('collection_date')['Rt'].quantile(stat).to_frame().reset_index()
    rt_df['collection_date'] =  pd.to_datetime(rt_df['collection_date'])
    return rt_df

def make_output_df(dates, y):
    """
    Input: NumPy array of Dates and Output
    Output: A DataFrame of collection Date and output Rt
    """
    dates_df = pd.DataFrame(data=dates, columns=['collection_date'])
    y_df = pd.DataFrame(data=y, columns=['Rt'])
    dates_df.reset_index(inplace = True)
    y_df.reset_index(inplace=True)
    df = pd.concat([dates_df, y_df], axis = 1)
    df['collection_date'] = pd.to_datetime(df['collection_date'])
    return df

def plot_roc(ml_models, norm_X_test, y_test, output_dir, filename='test-roc.png'):
    """ Plots an ROC AUC Curve with 95% Confidence Intervals
    Arguments:
        ml_models {Dict} -- Model Name as Key, Pickle ML models as Value
        norm_X_test {NumPy Array} -- Preprocessed X test xalues
        y_test {NumPy Array} -- y test values
        output_dir {String} -- Output Path to save ROC Curve
    """
    plt.figure(figsize=(6,6))
    for name, model in ml_models.items():
        y_pred_prob = model.predict_proba(norm_X_test)
        fpr, tpr, a = delong.get_prediction_stats(y_test, y_pred_prob[:,1])
        ci = delong.compute_stats(0.95,y_pred_prob[:,1],y_test)
        plt.plot(fpr, tpr, lw=2, label= name + ' ROC curve - (area = %(a)0.2f) (%(left)0.2f, %(right)0.2f)' % {'a':a, 'left': ci[0], 'right': ci[1]})
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Linear ROC - Test Set')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, filename))


def MSE(y_true,y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse

def summary_stats_models(models, X_test, y_test, output_dir):
    mse_list = []
    best_hyper_params = []
    for name, model in models.items():
        mse = mean_squared_error(y_test, model.predict(X_test))
        #logger.info("MSE of " + name + ": " + str(mse))
        mse_list.append(mse)

        # Also store the Best Hyperparams for each model
        best_hyper_params.append(model.best_params_)
        #logger.info(str(models[i].best_params_))
        

    # Write out the MSE in a file format
    summary_df = pd.DataFrame(data = [mse_list], columns = model_names)
    summary_df.to_csv(os.path.join(output_dir, 'summary_stats.csv'), index=False)
    hyperparams_df = pd.DataFrame(data=[best_hyper_params], columns = model_names)
    hyperparams_df.to_csv(os.path.join(output_dir, 'hyperparams.csv'), index=False)

def shap_summary_models(models, features_processed, X_test, output_dir):
    """
    Plot the the test data agains the true data as a time series

    Parameters
    ----------
    models : dict
             Key[string]: Machine Learning Model Name
             Value[pickle.sklearn.model]: Machine Learning Model
    features_processed : list
            List of preprocessed feature names
    X_test : pd.DataFrame
             DataFrame of preprocessed X values
    output_dir : string
             Output directory filepath
    """
    for name, model in models.items():
        plt.figure(figsize = (20,20))
        best_model = model.best_estimator_
        try:
            explainer = shap.TreeExplainer(best_model)
        except:
            explainer = shap.LinearExplainer(best_model, X_test)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, features = X_test, show = False, plot_size = 'auto', feature_names = features_processed)
        plt.title('SHAP summary plot of ' + name)
        plt.savefig(os.path.join(output_dir, 'SHAP_summary_' +  str(name) + '.png'), bbox_inches='tight')

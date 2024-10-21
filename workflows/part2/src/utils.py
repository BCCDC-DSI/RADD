import pandas as pd
import numpy as np
import re
import os

def create_output_dirs(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def clean_data(data, compound_col='Compound'):
    # Clean the 'Compound' column by removing special characters
    data['Stripped Compound Name'] = data[compound_col].apply(lambda x: re.sub(r'[^A-Za-z0-9 ]+', '', str(x)))
    return data

def load_and_clean_data(file_path):
    """
    Format the File correctly

    Parameters
    ----------
        file_path: str
            File path to the raw file
    Returns
    -------
        data : pd.DataFrame
            A clean dataframe, ready for further feature engineering
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    header_line = None
    for i, line in enumerate(lines):
        if "Compound Name" in line:
            header_line = i
            break

    if header_line is None:
        raise ValueError("Header line with 'Compound Name' not found")

    # Read the data from the header line onward
    data = pd.read_csv(file_path, skiprows=header_line)
    data = clean_data(data)
    
    # Move the 'Compound' column to the first position
    columns = ['Compound'] + [col for col in data.columns if col != 'Compound']
    data = data[columns]
    return data

def create_long_metrics_df(models_list, metrics_list, metric_col_name, dataset_name):
    """
    Create a Long DataFrame for the Metrics of the pipeline

    Parameters
    ----------
        models_list : List
            The models to add to a models column
        metrics_list : List
            The metrics to add to a metrics column
        dataset_name : String
            The Dataset the metrics were made on (Train/Test)
    
    Returns
    -------
        df : DataFrame
            Long Format DataFrame
    """
    df = pd.DataFrame(columns=['Model', metric_col_name])
    df['Model'] = models_list
    df[metric_col_name] = metrics_list
    df['Dataset'] = dataset_name

    return df
import pandas as pd
import numpy as np
import re
import os

def create_output_dirs(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    # Clean the 'Compound' column by removing special characters
    data['Compound'] = data['Compound'].apply(lambda x: re.sub(r'[^A-Za-z0-9 ]+', '', str(x)))
    data.drop(columns=['Compound Name'], inplace=True)
    # Move the 'Compound' column to the first position
    columns = ['Compound'] + [col for col in data.columns if col != 'Compound']
    data = data[columns]
    return data

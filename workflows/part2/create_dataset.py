# This script prepares the SMILES data for the ML Pipeline
import pandas as pd
import numpy as np
import yaml
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import pubchempy as pcp
import sys
sys.path.append('src/')
import smiles
import utils

with open('config/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model_index = config['model_index']
model_y = config['model_y']
raw_data_dir = config['raw_data_dir']
raw_data_filename = config['raw_data_filename']
smiles_dict_filename = config['smiles_dict_filename']
output_dir = config['output_dir']
output_filename = config['output_filename']


# Load and apply rdkit SMILES method
#data = utils.load_and_clean_data(os.path.join(raw_data_dir, raw_data_filename))
data = pd.read_csv(os.path.join(raw_data_dir, raw_data_filename))
print(data.head())
# we only wish to retain the not missing prediction + no repeats
print(data.shape)
data.dropna(subset=[model_y], inplace=True)
print(data.shape)
data.drop_duplicates(subset=[model_index], inplace=True)
print(data.shape)

smiles_dict = smiles.load_smiles_dict(smiles_dict_filename)

if smiles_dict != None:
    # Function to check and replace values
    def replace_values(row, value_dict):
        key = row[model_index]
        if key in value_dict:
            if row['SMILES'] != value_dict[key]:
                return value_dict[key]
        return row['SMILES']

    # Apply the function to replace values if needed
    data['SMILES'] = data.apply(lambda row: replace_values(row, smiles_dict), axis=1)

data['SMILES'] = data.apply(
    lambda x: smiles.get_smiles_with_url(
        x[model_index],  # The compound name (from model_index column)
        smiles_dict,
        smiles_dict_filename,
        existing_value=x['SMILES']  # The existing value from the 'SMILES' column
    ),
    axis=1  # Apply the function row by row
)
print(data.head())

# check canonical SMILES 
data = smiles.canonicalize_smiles(data, 'SMILES')
# Save the data
data.to_csv(os.path.join(output_dir,output_filename), index=False)

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
import utils

with open('config/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

working_dir = config['working_dir']
raw_data_dir = config['raw_data_dir']
raw_data_filename = config['raw_data_filename']
output_dir = config['output_dir']
output_filename = config['output_filename']
def compound_name_to_smiles(compound):
    """
    Parameters
    ----------
        compound : String
            Compound Name
    Returns
    -------
        smiles : String
            SMILEs in str format
    """
    def name_to_smiles(name):
        try:
            compound = pcp.get_compounds(name, 'name')
            if compound:
                return compound[0].canonical_smiles
            else:
                return None
        except Exception as e:
            return None

    if isinstance(compound, pd.Series):
        return compound.apply(name_to_smiles)
    elif isinstance(compound, str):
        return name_to_smiles(compound)
    else:
        raise ValueError("Input must be a string or a pandas Series.")

def add_smiles_to_data(data, compcound_col = 'Compound'):
    """
    Parameters
    ----------
        data : pd.DataFrame
            Dataframe with a compound column (default: `Column`)
    Returns
    -------
        data : pd.DataFrame
            The same dataframe with a  `SMILES` column         
    """
    # Extract unique compound names
    unique_compounds = data[compcound_col].unique()


    # Apply the SMILES conversion function to unique compound names
    smiles_dict = {compound: compound_name_to_smiles(compound) for compound in unique_compounds}

    # Map the SMILES strings back to the original DataFrame
    data['SMILES'] = data[compcound_col].map(smiles_dict)

    return data

# Read the training file
data = utils.load_and_clean_data(os.path.join(working_dir, raw_data_dir, raw_data_filename))
print(data.head())
data = add_smiles_to_data(data)
print(data.head())
utils.create_output_dirs(os.path.join(working_dir,output_dir))
data.to_csv(os.path.join(working_dir,output_dir,output_filename), index=False)

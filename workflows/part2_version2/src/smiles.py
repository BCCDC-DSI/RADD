import pandas as pd
import numpy as np
import re
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import pubchempy as pcp
import yaml
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

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

def add_smiles_to_data(data, compound_col = 'Compound'):
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
    unique_compounds = data[compound_col].unique()

    # Apply the SMILES conversion function to unique compound names
    smiles_dict = {compound: compound_name_to_smiles(compound) for compound in unique_compounds}

    # Map the SMILES strings back to the original DataFrame
    data['SMILES'] = data[compound_col].map(smiles_dict)

    return data

def get_smiles_with_url(compound_name, smiles_dict, pickle_filename):
    # Check if the compound's SMILES is already in the dictionary
    if compound_name in smiles_dict:
        return smiles_dict[compound_name]
    
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/property/CanonicalSMILES/TXT"
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    try:
        response = session.get(url)
        response.raise_for_status()
        smiles = response.text.strip()
        if smiles:
            smiles_dict[compound_name] = smiles
            save_smiles_dict(smiles_dict, pickle_filename)
            return smiles
        else:
            print(f"No SMILES data found for compound: {compound_name}")
            return None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred for {compound_name}: {http_err}")
    except Exception as err:
        print(f"An error occurred for {compound_name}: {err}")
    return None

# Function to save the SMILES dictionary to a pickle file
def save_smiles_dict(smiles_dict, pickle_filename):
    with open(pickle_filename, 'wb') as f:
        pickle.dump(smiles_dict, f)

# Function to load the SMILES dictionary from a pickle file
def load_smiles_dict(pickle_filename):
    if os.path.exists(pickle_filename):
        with open(pickle_filename, 'rb') as f:
            return pickle.load(f)
    else:
        return {}
    
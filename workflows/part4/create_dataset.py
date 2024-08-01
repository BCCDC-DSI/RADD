import pandas as pd
import os
import numpy as np
import pyproj
import yaml
import datetime

# Read the directory which houses the dat
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Load the configurations
data_dir = config['data_dir']
output_dir = config['output_dir']

# Read the DataFrame
df = pd.read_csv(data_dir)

print(df.head())

# Filter the periods
periods = ['e) Trough 3', 'f) Wave 3', 'g) Trough 4', 'h) Wave 4', 'i) Trough 5', 'j) Wave 5', 'k) Trough 6', 'l) Wave 6']
df = df[df['period'].isin(periods)]

# Convert collection date to datetime
df['collection_date'] = pd.to_datetime(df['collection_date'])

# Drop observations with death
df = df[df['death_date'].isna()]

# Fix Vax Status to categorical
df['vax_stat_row'] = df['vax_stat_row'].astype(str)

# List of features
features = ['collection_date','age', 'patient_phys_pc_HA', 'vax_stat_row', 'Ethno_Cultural_Composition_Score','Economic_dependency_Scores',
            'Ct_value', 'chsa_name', 'lineage', 'vaccine_type',
           'Residential_instability_Scores', 'Situational_Vulnerability_Scores', 'median_rt','delta_time_btw_vax_cat', 'hospitalized_4cov', 'episode_reinfection']    

# Select the features (**NOTE**: Includes the predictor!)
dataset = df[features]

# Wait for fix here
dataset.dropna(subset=['collection_date'], inplace=True)
dataset['episode_reinfection'] = dataset['episode_reinfection'].astype(bool)
# Write to a data directory
dataset['delta_time_btw_vax_cat'] = dataset['delta_time_btw_vax_cat'].str.replace('>', 'more than ')
dataset['delta_time_btw_vax_cat'] = dataset['delta_time_btw_vax_cat'].str.replace('<', 'less than ')

print(dataset.head())
print(dataset.info())
dataset.to_csv(os.path.join(output_dir, 'dataset.csv'), index=False)

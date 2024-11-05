import re
import pandas as pd
import numpy as np
import pymzml
import os
from tqdm import tqdm

import sys
print( sys.argv )

# Function to process each mzML file and extract relevant m/z data
def process_mzml_file(year, filename, relevant_mz_values, compound_name):
    file_path = os.path.join(mzml_dir, year, f"{filename}.mzML")
    data = []

    if not os.path.isfile(file_path):
        print(f"File {file_path} not found.")
        return pd.DataFrame(data, columns=['Compound Name', 'Spectrum_ID', 'Retention_Time', 'Transformed mass with error', 'm/z', 'Intensity'])

    # Read mzML file
    run = pymzml.run.Reader(file_path)
    for spectrum in run:
        # print( spectrum )
        if spectrum.ms_level == 1:  # Only process MS1 spectra
            retention_time = spectrum.scan_time_in_minutes()
            spectrum_id = spectrum.ID
            transformed_mass_with_error = spectrum.transformed_mz_with_error

            # Extract only relevant m/z peaks
            for mz, intensity in spectrum.peaks('raw'):
                if any(abs(mz - target_mz) < 0.01 for target_mz in relevant_mz_values):  # tolerance set to 0.01
                    data.append([compound_name, spectrum_id, retention_time, transformed_mass_with_error, mz, intensity])

    # Return as DataFrame
    return pd.DataFrame(data, columns=['Compound Name', 'Spectrum_ID', 'Retention_Time', 'Transformed mass with error', 'm/z', 'Intensity'])

project_folder = '/arc/project/st-ashapi01-1/git/afraz96/RADD/workflows/part4'
data_dir = 'Data'
job_outdir = "/scratch/st-ashapi01-1/part4_data"

db_filename = 'ms_library_20241011.csv'
prediction_filename = 'validity_data_summary_20241011.xlsx'

db_df = pd.read_csv(os.path.join(project_folder, data_dir, db_filename), skiprows=5)
prediction_df = pd.read_excel(os.path.join(project_folder, data_dir, prediction_filename),  engine='openpyxl')


# Clean up 'Compound Name' in file1_df
prediction_df['drug'] = prediction_df['drug'].str.lower().replace(r'[^a-z0-9]', '', regex=True).str.strip()

# Clean up 'Compound Name' in file2_df
db_df['Compound Name'] = db_df['Compound Name'].str.lower().replace(r'[^a-z0-9]', '', regex=True).str.strip()

# Step 2: Filter File 2 based on Compounds from File 1
compounds_from_file1 = prediction_df['drug'].unique()
file2_filtered = db_df[db_df['Compound Name'].isin(compounds_from_file1)]
# Create a set of filenames from prediction_df without extensions
prediction_filenames = set(prediction_df['file_name'].str.replace('.mzml', '', regex=False))
prediction_df['file_name'] = prediction_df['file_name'].str.replace('.mzml', '')
# Main processing loop for each file in prediction_df
all_data = []
extra_files_count = 0
total_files_count = 0
mz_values_from_file2 = file2_filtered['m/z'].unique()

# Specify the path to your mzML file
mzml_dir = '/arc/project/st-ashapi01-1/bccs_mzml'
try:
    years = [np.int8( sys.argv[1] )]  
except:
    years = [ 20,21,22,23,24]

print( '\n\nInput argument:', years )

# Process each filename in the mzML directory by iterating over years
for year in years:
    year_dir = os.path.join(mzml_dir, '%s'%(2000 + year) )
    if os.path.isdir(year_dir):
        L= os.listdir(year_dir)  
        print( len(L), 'files for year', year )
        for i, filename in enumerate( L ):
            if filename.endswith('.mzML'):
                total_files_count += 1
                base_filename = filename.replace('.mzML', '')  # Strip extension

                # Check if this file is in prediction_df
                if base_filename not in prediction_filenames:
                    extra_files_count += 1  # Count as extra if not in prediction_df
                    continue

                # Filter for matching row
                prediction_rows = prediction_df[prediction_df['file_name'] == base_filename]
                if prediction_rows.empty:
                    extra_files_count += 1
                    continue  # Skip to the next file if no match is found

                if prediction_rows.shape[0] != 21:
                    print( i, prediction_rows.shape[0] )
                # Extract compound name and process file if it's in prediction_df
                for row in tqdm( range(prediction_rows.shape[0]) ):
                    prediction_row = prediction_rows.iloc[row,: ]
                    compound_name = prediction_row['drug']
                    y = prediction_row['detection_status'] == 'detected'
                    relevant_mz_values = file2_filtered[file2_filtered['Compound Name'] == compound_name]['m/z'].values
    
                    # Process the mzML file and extract relevant data
                    e = process_mzml_file(year, base_filename, relevant_mz_values, compound_name)
                    e.to_csv( os.path.join(  job_outdir, f'{base_filename}_{compound_name}_{y}.csv'), index=False)                
                    print( i, e.shape[0], end=' ', flush=True ) # for progress update
                    
                if 0:#extracted_data is not None:
                    try:
                        all_data.append(extracted_data)
                    except Exception as e:
                        print(e)                
                        
                if 0:#(i % 100 )==99:
                    print( 'saving after processing', i, 'for', year )
                    final_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
                    final_df.to_csv(os.path.join( job_outdir, 'filtered_mzml_data.csv'), index=False)

if 0:
    # Combine all extracted data into a single DataFrame
    final_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    # Calculate and print the extra file ratio
    extra_file_ratio = extra_files_count / total_files_count if total_files_count > 0 else 0
    print(f"Total mzML files in directory: {total_files_count}")
    print(f"Extra files not found in prediction_df: {extra_files_count}")
    print(f"Extra file ratio: {extra_file_ratio:.2%}")
    
    # Save or link final_df with prediction_df as needed
    final_df.to_csv(os.path.join( job_outdir, 'filtered_mzml_data.csv'), index=False)
    print("Data successfully extracted and saved to filtered_mzml_data.csv")


# Process each filename in the mzML directory by iterating over years
# Process each filename in the mzML directory by iterating over years
for year in years:
    year_dir = os.path.join(mzml_dir, year)
    if os.path.isdir(year_dir):
        for i, filename in enumerate( os.listdir(year_dir) ):
            if filename.endswith('.mzML'):
                total_files_count += 1
                base_filename = filename.replace('.mzML', '')  # Strip extension

                # Check if this file is in prediction_df
                if base_filename not in prediction_filenames:
                    extra_files_count += 1  # Count as extra if not in prediction_df
                    continue

                # Filter for matching row
                prediction_row = prediction_df[prediction_df['file_name'] == base_filename]
                if prediction_row.empty:
                    extra_files_count += 1
                    continue  # Skip to the next file if no match is found

                # Extract compound name and process file if it's in prediction_df
                prediction_row = prediction_row.iloc[0]
                compound_name = prediction_row['drug']
                relevant_mz_values = file2_filtered[file2_filtered['Compound Name'] == compound_name]['m/z'].values

                # Process the mzML file and extract relevant data
                extracted_data = process_mzml_file(year, base_filename, relevant_mz_values, compound_name)
                if extracted_data is not None:
                    all_data.append(extracted_data)
                    print(all_data)

                if (i % 1000 )==0:
                    print( 'saving after processing', i, 'for', year )   
                    final_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
                    final_df.to_csv(os.path.join(project_folder, data_dir, 'filtered_mzml_data.csv'), index=False)


# Combine all extracted data into a single DataFrame
final_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# Calculate and print the extra file ratio
extra_file_ratio = extra_files_count / total_files_count if total_files_count > 0 else 0
print(f"Total mzML files in directory: {total_files_count}")
print(f"Extra files not found in prediction_df: {extra_files_count}")
print(f"Extra file ratio: {extra_file_ratio:.2%}")

# Save or link final_df with prediction_df as needed
final_df.to_csv(os.path.join(project_folder, data_dir, 'filtered_mzml_data.csv'), index=False)
print("Data successfully extracted and saved to filtered_mzml_data.csv")

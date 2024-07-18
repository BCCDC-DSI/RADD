# Define variables for file paths
SCRIPT_PATH="/arc/project/st-ashapi01-1/git/afraz96/RADD/workflows/part2_version2/generate_model_stats.py"
OUTPUT_PATH="/scratch/st-ashapi01-1/RADD/SMILES_ML_PIPELINE/X500R_output"
DATA_PATH="/arc/project/st-ashapi01-1/git/afraz96/RADD/workflows/part2_version2/Data/training_data_bccdc_20240708.csv"
TEST_DATA_PATH="/arc/project/st-ashapi01-1/git/afraz96/RADD/workflows/part2_version2/Data/X500R_SMILES.csv"
MODELS_PATH="/scratch/st-ashapi01-1/RADD/SMILES_ML_PIPELINE/X500R_output/models"
PROCESSOR_PATH="/scratch/st-ashapi01-1/RADD/SMILES_ML_PIPELINE/X500R_output/processor.pkl"
# Print the date and time
echo "Job started at: $(date)"

# Run the Python script with specified arguments
python $SCRIPT_PATH -o $OUTPUT_PATH -d $TEST_DATA_PATH -t $DATA_PATH -m $MODELS_PATH -p $PROCESSOR_PATH

# Print the date and time
echo "Job finished at: $(date)"
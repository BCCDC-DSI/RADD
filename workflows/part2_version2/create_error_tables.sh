# Define variables for file paths
SCRIPT_PATH="/arc/project/st-ashapi01-1/git/afraz96/RADD/workflows/part2_version2/generate_model_stats.py"
OUTPUT_PATH="/scratch/st-ashapi01-1/RADD/SMILES_ML_PIPELINE/NPS_OUTPUT"
DATA_PATH="/arc/project/st-ashapi01-1/git/afraz96/RADD/workflows/part2_version2/Data/training_data_bccdc_20240708.csv"
TEST_DATA_PATH="/arc/project/st-ashapi01-1/git/afraz96/RADD/workflows/part2_version2/Data/X500R_SMILES.csv"
MODELS_PATH="/scratch/st-ashapi01-1/RADD/SMILES_ML_PIPELINE/NPS_OUTPUT/models"
PROCESSOR_PATH="/scratch/st-ashapi01-1/RADD/SMILES_ML_PIPELINE/NPS_OUTPUT/processor.pkl"
# Print the date and time
echo "Job started at: $(date)"

# Run the Python script with specified arguments
python $SCRIPT_PATH -o $OUTPUT_PATH -d $DATA_PATH -t $TEST_DATA_PATH -m $MODELS_PATH -p $PROCESSOR_PATH

# Print the date and time
echo "Job finished at: $(date)"
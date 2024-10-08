#!/bin/sh
#SBATCH --job-name=ml_pipeline         # Job name
#SBATCH --account=st-ashapi01-1        # Specify your allocation code
#SBATCH --job-name=smiles_ml_pipeline  # Specify the job name
#SBATCH --ntasks=8                     # Number of tasks (processes)
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --time=140:00:00               # Time limit hrs:min:sec
#SBATCH --mem=8G                       # Memory limit
#SBATCH --array=1-5                    # Job array
#SBATCH --output=logs/array_%A_%a.out  # Redirects standard output to unique files for each sub-job.
#SBATCH --error=logs/array_%A_%a.err   # Redirects standard error to unique files for each sub-job.
#SBATCH --partition=skylake            # Partition name
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
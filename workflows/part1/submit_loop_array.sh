#!/bin/bash
#SBATCH --time=30:00:00                    # Request 3 hours of runtime
#SBATCH --account=st-ashapi01-1            # Specify your allocation code
#SBATCH --job-name=radd-p1              # Specify the job name
#SBATCH --nodes=1                       # Defines the number of nodes for each sub-job.
#SBATCH --ntasks-per-node=1             # Defines tasks per node for each sub-job.
#SBATCH --mem=32G                        # Request 8 GB of memory
#SBATCH --array=3-5 
#SBATCH --error=%A_%a.err
##  SBATCH --mail-user=lisa.tang@bccdc.ca   # Email address for job notifications
##  SBATCH --mail-type=ALL                 # Receive email notifications for all job events

export LC_ALL=C; unset LANGUAGE

output_dir=$3

# Load necessary modules
module load gcc
module load apptainer

# Create output folder if not already created
if [ ! -d ${output_dir} ]; then
  mkdir -p $output_dir;
fi

# Compile the command 
cmd="/arc/project/st-ashapi01-1/RADD/library_final/apptainer/ubuntu.sandbox Rscript /arc/project/st-ashapi01-1/RADD/workflow1/inner_script.R ${1} ${2} ${output_dir} ${SLURM_ARRAY_TASK_ID} ${4}"
echo -e "Command: $cmd"

# Call with the cmd which executes inner.R script that will loop over files found by system call in R

if [ "$#" -eq 4 ]; then
  apptainer exec --nv $cmd
else
  echo -e "Error: [mzML_folder] [database_csv] [output_path] [number_of_batches] [A] [B] \nAll files will be divided into N job arrays N=[number_of_batches].\nThe current sbatch call will queue arrays starting from A to B "
  set -e
fi





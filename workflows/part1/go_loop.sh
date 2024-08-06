#!/bin/bash
#SBATCH --time=30:00:00                 # Request 3 hours of runtime
#SBATCH --account=st-ashapi01-1         # Specify your allocation code
#SBATCH --job-name=radd-p1              # Specify the job name
#SBATCH --nodes=1                       # Defines the number of nodes for each sub-job.
#SBATCH --ntasks-per-node=1             # Defines tasks per node for each sub-job.
#SBATCH --mem=32G                       # Request 8 GB of memory
#SBATCH --array=1-50 
#SBATCH --error=%A_%a.err

export LC_ALL=C; unset LANGUAGE

output_dir=$3

# Load necessary modules
module load gcc
module load apptainer

script_path='/arc/project/st-ashapi01-1/git/ywtang/RADD/workflows/part1/go_loop.R'

# Compile the command 
cmd="/arc/project/st-ashapi01-1/RADD/library_final/apptainer/ubuntu.sandbox Rscript ${script_path}"
cmd="${cmd} ${1} ${2} ${output_dir} ${SLURM_ARRAY_TASK_ID} ${4} ${5}"
echo -e "Command: $cmd"

# Call with the cmd which executes inner.R script that will loop over files found by system call in R
if [ "$#" -eq 5 ]; then
  apptainer exec --nv $cmd
else
  echo -e "Error in the sbatch argument list: [mzML_folder] [database_csv] [output_path] [number_of_batches] [VERBOSE] \nAll files will be divided into N job arrays N=[number_of_batches]."
  set -e
fi

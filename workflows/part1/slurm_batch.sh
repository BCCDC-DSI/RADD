#!/bin/bash
#SBATCH --time=3:00:00                    # Request 3 hours of runtime
#SBATCH --account=st-cfjell-1            # Specify your allocation code
#SBATCH --job-name=nps_job_array         # Specify the job name
#SBATCH --nodes=1                       # Defines the number of nodes for each sub-job.
#SBATCH --ntasks-per-node=1             # Defines tasks per node for each sub-job.
#SBATCH --mem=8G                        # Request 8 GB of memory
#SBATCH --output=logs/array_%A_%a.out        # Redirects standard output to unique files for each sub-job.
#SBATCH --error=logs/array_%A_%a.err         # Redirects standard error to unique files for each sub-job.
#SBATCH --mail-user=cfjell@mail.ubc.ca   # Email address for job notifications
#SBATCH --mail-type=ALL                 # Receive email notifications for all job events
export LC_ALL=C; unset LANGUAGE

# get list of input files, input directory is the first passed-in parameter
input_dir=$1
input_files=($input_dir/*mzML)
input_file=${input_files[$SLURM_ARRAY_TASK_ID]}

#Load necessary modules
module load gcc
module load apptainer

#Change to working directory
apptainer exec --nv /arc/project/st-cfjell-1/apptainer/ubuntu.sandbox Rscript /arc/project/st-cfjell-1/git/mass_spec/go.R $input_file

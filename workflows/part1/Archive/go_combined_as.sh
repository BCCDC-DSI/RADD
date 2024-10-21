#!/bin/bash
#SBATCH --time=30:00:00                    # Request 3 hours of runtime
#SBATCH --account=st-ashapi01-1            # Specify your allocation code
#SBATCH --job-name=radd-p1-2024              # Specify the job name
#SBATCH --nodes=1                       # Defines the number of nodes for each sub-job.
#SBATCH --ntasks-per-node=1             # Defines tasks per node for each sub-job.
#SBATCH --mem=32G                        # Request 8 GB of memory
#SBATCH --array=1-5 
#SBATCH --error=%A_%a.err
##  SBATCH --mail-user=aaron.shapiro1@phsa.ca   # Email address for job notifications
##  SBATCH --mail-type=ALL                 # Receive email notifications for all job events

# Create environment variable LC_ALL and unset LANGUAGE
export LC_ALL=C
unset LANGUAGE

# Load required modules
module load gcc
module load apptainer

# Navigate to the Apptainer directory
cd /arc/project/st-ashapi01-1/RADD/library_final/apptainer

# Bind directories and start the Apptainer shell with the sandbox image
apptainer shell --bind /arc/software,/arc/home,/arc/project,/scratch /arc/project/st-ashapi01-1/RADD/library_final/apptainer/ubuntu.sandbox <<EOF

# Change directory after entering the container
cd /arc/project/st-ashapi01-1/RADD

# Start R
R --no-save <<R_SCRIPT

# Update the path to use XCMS 3.2.2 instead of 4.2.0
.libPaths(c('/arc/project/st-ashapi01-1/RADD/library_final/R/x86_64-pc-linux-gnu-library/4.3', .libPaths()))

# Load and run the R pipeline
source('/arc/project/st-ashapi01-1/RADD/workflow1/go_loop_combined_20240920.R')

R_SCRIPT
EOF

#!/bin/bash
#SBATCH --time=30:00:00                 # Request 3 hours of runtime
#SBATCH --account=st-ashapi01-1         # Specify your allocation code
#SBATCH --job-name=RADD-rds2csv         # Specify the job name
#SBATCH --nodes=1                       # Defines the number of nodes for each sub-job.
#SBATCH --ntasks-per-node=1             # Defines tasks per node for each sub-job.
#SBATCH --mem=32G                       # Request 8 GB of memory
#SBATCH --array=1 
#SBATCH --error=%A_%a.err

export LC_ALL=C; unset LANGUAGE

output_dir=$3

# Load necessary modules
module load gcc
module load apptainer

script_path='/arc/project/st-ashapi01-1/git/ywtang/RADD/workflows/part1/rds_to_csv.R'

cmd="/arc/project/st-ashapi01-1/RADD/library_final/apptainer/ubuntu.sandbox Rscript ${script_path}"
cmd="${cmd} /scratch/st-ashapi01-1/expedited_2020/combined_db_20240801"
echo -e "Command: ${cmd}"
apptainer exec --nv ${cmd}

cmd="/arc/project/st-ashapi01-1/RADD/library_final/apptainer/ubuntu.sandbox Rscript ${script_path}"
cmd="${cmd} /scratch/st-ashapi01-1/expedited_2021/combined_db_20240801"
echo -e "Command: ${cmd}"
apptainer exec --nv ${cmd}

cmd="/arc/project/st-ashapi01-1/RADD/library_final/apptainer/ubuntu.sandbox Rscript ${script_path}"
cmd="${cmd} /scratch/st-ashapi01-1/expedited_2022/combined_db_20240801"
echo -e "Command: ${cmd}"
apptainer exec --nv ${cmd}

cmd="/arc/project/st-ashapi01-1/RADD/library_final/apptainer/ubuntu.sandbox Rscript ${script_path}"
cmd="${cmd} /scratch/st-ashapi01-1/expedited_2023/combined_db_20240801"
echo -e "Command: ${cmd}"
apptainer exec --nv ${cmd}

cmd="/arc/project/st-ashapi01-1/RADD/library_final/apptainer/ubuntu.sandbox Rscript ${script_path}"
cmd="${cmd} /scratch/st-ashapi01-1/expedited_2024/combined_db_20240801"
echo -e "Command: ${cmd}"
apptainer exec --nv ${cmd}



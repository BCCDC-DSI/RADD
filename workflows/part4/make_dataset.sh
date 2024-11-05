#!/bin/sh
#SBATCH --job-name=output_indv_raddpt4                                      # Job name
#SBATCH --account=st-ashapi01-1                                             # Specify your allocation code (add -gpu in front of the account)
#SBATCH --ntasks=1                                                          # Number of tasks (processes)
#SBATCH --nodes=1                                                           # Number of nodes
#SBATCH --time=140:00:00                                                    # Time limit hrs:min:sec
#SBATCH --mem=16G                                                           # Memory limit
#SBATCH --array=20-24                                                       # Job array; number cannot be too big
#SBATCH --error=/scratch/st-ashapi01-1/part4_logs_ywtang/make_dataset_%A_%a.err         # Redirects standard error to unique files for each sub-job.
#SBATCH --partition=skylake                                                             # Partition name

## Print the date and time
echo "Job started at: $(date)"

## init conda
#export HOME=/scratch/st-ashapi01-1/
#conda config --set auto_activate_base true
#conda init
#conda activate /home/ywtang/miniconda3/envs/chemenv

##activate VE
ODIR=/scratch/st-ashapi01-1/
source ${ODIR}/raddpt4/bin/activate

## Run the Python script with specified arguments
SCRIPT_PATH=/arc/project/st-ashapi01-1/git_ssh/ywtang/RADD/workflows/part4/create_dataset_every1000.py
python $SCRIPT_PATH ${SLURM_ARRAY_TASK_ID}

## Print the date and time
echo "Job finished at: $(date)"


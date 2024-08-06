#!/bin/bash

# This script is meant to run interactively, **not** as a SLURM job submission
#
# cd /scratch/
# bash rds_to_csv.sh rds_location_absolute_path 

module load gcc/5.5.0 r/4.4.0

Rscript rds_to_csv.R $1 > $1.log 

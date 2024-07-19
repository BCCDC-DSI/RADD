#!/bin/bash

module load gcc/5.5.0 r/4.4.0

Rscript rds_to_csv.R $1 > $1.log 

#!/bin/bash

module load gcc/5.5.0 r/4.4.0

Rscript rds_to_csv.R $1 > $2.log 

echo -e "**Repeat for each cohort when the script finishes**

Please run something like following:

cd data_highresnps/ms1/
head -n 1 2023-0001BG01_ms1.csv > combined_ms1.csv && tail -n+2 -q *.csv >> combined_ms1.csv

cd data_highresnps/ms2
head -n 1 2023-0001BG01_ms2.csv > combined_ms2.csv && tail -n+2 -q *.csv >> combined_ms2.csv
" 

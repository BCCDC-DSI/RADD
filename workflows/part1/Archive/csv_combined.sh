#!/bin/bash


cd /scratch/st-ashapi01-1/rds_files/data_2024nps-db/ms1/
head -n 1 2023-0001BG01_ms1.csv > combined.txt && tail -n+2 -q *.csv >> combined.txt

mv combined.txt combined_ms1.csv

current_date_time="`date +%Y%m%d%H%M%S`";
echo -e "Finished joining CSV at $current_date_time"



cd /scratch/st-ashapi01-1/rds_files/data_2024nps-db/ms2/
head -n 1 2023-0001BG01_ms2.csv > combined.txt && tail -n+2 -q *.csv >> combined.txt
mv combined.txt combined_ms2.csv

current_date_time="`date +%Y%m%d%H%M%S`";
echo -e "Finished joining CSV at $current_date_time"



cd /scratch/st-ashapi01-1/rds_files/data_highresnps/ms1/
head -n 1 2023-0001BG01_ms1.csv > combined.txt && tail -n+2 -q *.csv >> combined.txt
mv combined.txt combined_ms1.csv

current_date_time="`date +%Y%m%d%H%M%S`";
echo -e "Finished joining CSV at $current_date_time"



cd /scratch/st-ashapi01-1/rds_files/data_highresnps/ms2/
head -n 1 2023-0001BG01_ms2.csv > combined.txt && tail -n+2 -q *.csv >> combined.txt 
mv combined.txt combined_ms2.csv

current_date_time="`date +%Y%m%d%H%M%S`";
echo -e "Finished joining CSV at $current_date_time"


# To see creation times
# ls -ls /scratch/st-ashapi01-1/rds_files/data_*/ms*/combined_ms*.csv 


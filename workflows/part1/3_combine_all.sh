#!/bin/bash


cd /scratch/st-ashapi01-1/expedited_2020/combined_db_20240801/  
file=2020-1640BG01
head -n 1 ${file}_ms1.csv > combined_ms1.txt && tail -n+2 -q *ms1.csv >> combined_ms1.txt
head -n 1 ${file}_ms2.csv > combined_ms2.txt && tail -n+2 -q *ms2.csv >> combined_ms2.txt

cd /scratch/st-ashapi01-1/expedited_2024/combined_db_20240801
file=2024-0175BG01  
head -n 1 ${file}_ms1.csv > combined_ms1.txt && tail -n+2 -q *ms1.csv >> combined_ms1.txt
head -n 1 ${file}_ms2.csv > combined_ms2.txt && tail -n+2 -q *ms2.csv >> combined_ms2.txt

cd /scratch/st-ashapi01-1/expedited_2022/combined_db_20240801/
file=2022-0140BG01
head -n 1 ${file}_ms1.csv > combined_ms1.txt && tail -n+2 -q *ms1.csv >> combined_ms1.txt
head -n 1 ${file}_ms2.csv > combined_ms2.txt && tail -n+2 -q *ms2.csv >> combined_ms2.txt




cd /scratch/st-ashapi01-1/expedited_2023/combined_db_20240801/
file=2023-0001BG01
head -n 1 ${file}_ms1.csv > combined_ms1.txt && tail -n+2 -q *ms1.csv >> combined_ms1.txt
head -n 1 ${file}_ms2.csv > combined_ms2.txt && tail -n+2 -q *ms2.csv >> combined_ms2.txt

cd /scratch/st-ashapi01-1/expedited_2021/combined_db_20240801
file=2021-0301BG01   
head -n 1 ${file}_ms1.csv > combined_ms1.txt && tail -n+2 -q *ms1.csv >> combined_ms1.txt
head -n 1 ${file}_ms2.csv > combined_ms2.txt && tail -n+2 -q *ms2.csv >> combined_ms2.txt





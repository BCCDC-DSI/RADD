# Setup

```
module load intel-oneapi-compilers/2023.1.0 python/3.11.6
ODIR=/scratch/st-ashapi01-1/
python -m venv ${ODIR}/raddpt4
source ${ODIR}/raddpt4/bin/activate
pip3 install -r requirements.txt
```

## 1. Convert indv. mzmL to .csv 

```
sbatch /arc/project/st-ashapi01-1/git_ssh/ywtang/RADD/workflows/part4/make_dataset.sh
```

<details> 

<summary># of files</summary>


As of 2024-11-04, the number of samples by year:  
```
819 files for year 2020
1768 files for year 2021
1818 files for year 2022
2100 files for year 2023
1697 files for year 2024
```

```
2020: most searched for 21 compounds 
2021: most searched for 24 compounds
2022: most searched for 21 compounds
2023: most searched for 25 compounds
2024: most searched for 25 compounds
```

</details>

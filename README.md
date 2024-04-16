# RADD

## Getting started
1. Connect to ```myvpn.ubc.ca``` through myVPN
 
2. Connect to Sockey through terminal:
```
ssh username@sockeye.arc.ubc.ca
```

3. Navigate to project folder:
```
cd /arc/project/st-username-1/RADD
```

<details>

4a. Build virtual environment:
 
 ```
 cd $HOME
 
 wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
 bash Miniconda3-latest-Linux-x86_64.sh
 
 conda create --prefix /project/st-username-1/nps-screening/env
 conda activate /project/st-username-1/nps-screening/env
 conda create --name radd
 conda activate radd
 conda install R; conda install r-tidyverse r-magrittr r-argparse; conda install -c bioconda bioconductor-xcms
 ```

4b. Use apptainer(?)

</details>



## Tasks

| Tasks | Status |
| :-- | :-- |
| Sockeye: proposed new location: /arc/project/st-ashapi01-1/RADD| done|
| Sockeye: ask for symbolic links to: <br>/arc/project/st-cfjell-1/ms_data/expedited_2023/mzML<br>/arc/project/st-cfjell-1/apptainer| pending |
| Access to Chris' GitRepo | 50% |

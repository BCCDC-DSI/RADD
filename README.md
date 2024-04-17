# RADD


Notes:
- Sockeye [not accessible Apr 16, 2024]
- ComputeCanada [only accessible by Lisa]
 
## Getting started
1. Connect to ```myvpn.ubc.ca``` through myVPN
   - [Set up guide for Mac users](https://ubc.service-now.com/kb_view.do?sysparm_article=KB0017956#macos)
   - Window users may need to email and request OneDrive link to download an installer, as Lisa did in April, 2024
 
3. Connect to Sockey through terminal:
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

 conda create --name radd
 conda activate radd
 conda install R; conda install r-tidyverse r-magrittr r-argparse; conda install -c bioconda bioconductor-xcms
 ```
 
 
 Strategy 2: https://www.biostars.org/p/450316/  [tested on ComputeCanada 2024-04-16]
 
 ```
 cd $HOME
 
 wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
 bash Miniconda3-latest-Linux-x86_64.sh
 
 base_dir=$(echo $PWD)
 
 export PATH=$base_dir/miniconda/bin:$PATH
 source ~/.bashrc
 echo -e "$base_dir/miniconda/etc/profile.d/conda.sh" >> ~/.profile
 conda init bash
 
 # installing Mamba for fasta downloading of packages in conda
 conda install mamba -n base -c conda-forge -y
 conda update conda -y
 conda update --all
 
 # Creating R environment in conda
 mamba create -n R -c conda-forge r-base -y

 # Activating R environment
 conda activate R
 mamba install -c conda-forge r-essentials

 ```

4b. Use apptainer(?)


Note: To undo initialization of conda upon startup, issue:
```conda init --reverse $SHELL```

</details>



## Tasks

| Tasks | Status |
| :-- | :-- |
| Sockeye: proposed new location: ```/arc/project/st-ashapi01-1/RADD``` | request sent on 2024-04-16 |
| Sockeye: ask for symbolic links to: <br>```/arc/project/st-cfjell-1/ms_data/expedited_2023/mzML```<br>```/arc/project/st-cfjell-1/apptainer``` | pending |
| Access to Chris' GitRepo | 50% |
| Part 1: replicate analysis on data under ```/arc/project/st-cfjell-1/ms_data/expedited_2023/mzML``` | Steps 0 of n |

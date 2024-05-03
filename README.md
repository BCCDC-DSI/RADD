# RADD


Notes:
- Sockeye [only accessible by Afraz]
- ComputeCanada [only accessible by Lisa]
 
## One time setup 

1. Apply for Sockeye allocation: https://flex.redcap.ubc.ca/surveys/?s=7MKJT898LK
2. Setup [Multi-factor authentication](https://mfadevices.id.ubc.ca/). This is mandatory step or you will not be able to SSH.
3. Install myVPN:
   - [Set up guide for Mac users](https://ubc.service-now.com/kb_view.do?sysparm_article=KB0017956#macos)
   - Window users may need to email and request OneDrive link to download an installer, as Lisa did in April, 2024


## One time setup 

1. Connect to ```myvpn.ubc.ca``` through myVPN app
 
3. Connect to to Sockey through terminal using Secure Socket Shell protocol (SSH):
```
ssh username@sockeye.arc.ubc.ca
```

3. Navigate to project folder:
```
cd /arc/project/st-username-1/RADD
```

4a. Build virtual environment

 <details>
  
 Proposal 1: 
 ```
 cd $HOME
 
 wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
 bash Miniconda3-latest-Linux-x86_64.sh

 conda create --name radd
 conda activate radd
 conda install R; conda install r-tidyverse r-magrittr r-argparse; conda install -c bioconda bioconductor-xcms
 ```
 
 
 Proposal 2: https://www.biostars.org/p/450316/  [tested on ComputeCanada 2024-04-16]
 
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

 Note: To undo initialization of conda upon startup, issue:
 ```conda init --reverse $SHELL```
</details>

4b. Use apptainer prepared by Chris [see logs](https://github.com/BCCDC-DSI/RADD/blob/main/workflows/part1/log.md)



## Summary of tasks

| Tasks | Status |
| :-- | :-- |
| Sockeye: proposed new location: ```/arc/project/st-ashapi01-1/RADD``` | request sent on 2024-04-16 |
| Sockeye: ask for symbolic links to: <br>```/arc/project/st-cfjell-1/ms_data/expedited_2023/mzML```<br>```/arc/project/st-cfjell-1/apptainer``` | not needed |
| Access to Chris' GitRepo | 50% |
| Part 1: replicate analysis on data under ```/arc/project/st-cfjell-1/ms_data/expedited_2023/mzML``` | Steps 1 of 5 |

<details>
<summary>Part 1 </summary>
 
- [ ] working script from start to end
- [ ] outputs available for part 2
- [ ] ...

</details>



<details>
<summary>Part 2 </summary>
 
TBD

</details>

# Work in progress for LC-MS/MS
Based on workflows from R xcms library, as in https://www.rformassspectrometry.org/

### Install packages
To install all the packages in https://www.rformassspectrometry.org/pkgs/
Couldn't do it in a conda env. Also needed current R 4.3


Ubuntu 20
`sudo apt update`
`sudo apt upgrade`
`sudo apt install libssl-dev curl libcurl4-openssl-dev  pkg-config libxml2 xml2 libxml2-dev netcdf-bin libnetcdf-dev libglpk-dev libglpk40 libigraph-dev`


`sudo apt install libssl-dev curl libcurl4-openssl-dev  pkg-config libxml2 xml2 libxml2-dev netcdf-bin libnetcdf-dev libglpk-dev libglpk40 libigraph-dev`

```
# install R following at https://mirror.rcg.sfu.ca/mirror/CRAN/
# update indices
sudo apt update -qq
# install two helper packages we need
sudo apt install --no-install-recommends software-properties-common dirmngr
# add the signing key (by Michael Rutter) for these repos
# To verify key, run gpg --show-keys /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc 
# Fingerprint: E298A3A825C0D65DFD57CBB651716619E084DAB9
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
# add the R 4.0 repo from CRAN -- adjust 'focal' to 'groovy' or 'bionic' as needed
sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
###
sudo apt install --no-install-recommends r-base r-base-dev
#### end of R installation 

## install RforMassSpec packages (one at a time)
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("xcms")
BiocManager::install("MsExperiment")
BiocManager::install("igraph")
BiocManager::install(c("Spectra", "PSMatch", "MsCoreUtils", "MetaboCoreUtils"))

```

# Port of nps-screening to sockeye
**unsuccessful**

on Sockeye, 
1. in `.bashrc`, set `export R_LIBS_USER="/arc/project/st-cfjell-1/R_LIBS_USER"`
2. `module load R`
3. `conda create --name nps-screening`
4. `conda install netcdf4 libnetcdf`
5. `conda install glpk`

# Apptainer steps on sockeye
1. cd /arc/project/st-cfjell-1/apptainer
2. module load gcc apptainer
3. apptainer build --sandbox ubuntu.sandbox docker://ubuntu
4. apptainer shell --writable ubuntu.sandbox/
5. mkdir /ms_nps_screening
6. HOME=/ms_nps_screening
7. apt update
8. apt upgrade ## FAILED

# another try with binary starter
1. mkdir bioc_apptainer
2. cd bioc_apptainer
3. module load gcc apptainer
4. apptainer pull --name bioconductor.sif docker://bioconductor/bioconductor_docker
   4. [cfjell@login01 bioc_apptainer]$ ll
   5. total 1811088
   6. -rwxrwx--- 1 cfjell st-cfjell-1-rw 1479532544 Feb 13 18:17 bioconductor.sif
5. apptainer shell bioconductor.sif
6. R
   6. BiocManager::install("xcms")


# Create local apptainer then move to sockeye
See https://apptainer.org/docs/admin/main/installation.html#install-ubuntu-packages
On local WSL:
1. sudo add-apt-repository -y ppa:apptainer/ppa
2. sudo apt update
3. sudo apt install -y apptainer
4. mkdir ubuntu_nps_scr; cd ubuntu_nps_scr
4. apptainer build --sandbox ubuntu.sandbox docker://ubuntu
5. sudo apptainer shell --writable ubuntu.sandbox
6. apt install libssl-dev curl libcurl4-openssl-dev  pkg-config libxml2 xml2 libxml2-dev netcdf-bin libnetcdf-dev libglpk-dev libglpk40 libigraph-dev
7. sudo apt install --no-install-recommends software-properties-common dirmngr
8. apt install wget
9. wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
10. add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
11. apt install --no-install-recommends r-base r-base-dev
12. apt install r-cran-systemfonts r-cran-textshaping libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev
12. R
    13. if (!require("BiocManager", quietly = TRUE))
        install.packages("BiocManager")
    14. BiocManager::install("xcms")
    15. BiocManager::install("MsExperiment")
    16. BiocManager::install("igraph")
    17. BiocManager::install(c("Spectra", "PSMatch", "MsCoreUtils", "MetaboCoreUtils"))
    18. install.packages(c("argparse", "tidyverse", "magrittr"))

Tar and scp apptainer dir to sockeye 
```shell
scp .\apptainer\ubuntu.sandbox.tgz cfjell@sockeye.arc.ubc.ca:/arc/project/st-cfjell-1/apptainer
module load gcc
module load apptainer

(base) [cfjell@login03 apptainer]$ apptainer shell apptainer shell --writable^C
(base) [cfjell@login03 apptainer]$ apptainer shell --writable ubuntu.sandbox
WARNING: Skipping mount /arc/software [binds]: /arc/software doesn't exist in container
## Slurm execution is from shell that calls apptainer
WARNING: Skipping mount /arc/home [binds]: /arc/home doesn't exist in container
WARNING: Skipping mount /arc/project [binds]: /arc/project doesn't exist in container
WARNING: Skipping mount /scratch [binds]: /scratch doesn't exist in container
```
Need to add bindings
```shell
apptainer shell --bind /arc/software,/arc/home,/arc/project,/scratch ubuntu.sandbox
```




# Slurm
Slurm example batch shell file from :
:
```shell
#!/bin/bash
 
#SBATCH --time=xx:00:00        # Request xx hours of runtime
#SBATCH --account=alloc-code-gpu   # Specify your allocation code (with -gpu appended!)
#SBATCH --nodes=x              # Request x nodes
#SBATCH --gpus-per-node=x      # Request x GPUs per node
#SBATCH --mem=xG               # Request x GB of memory
#SBATCH --job-name=job_name    # Specify the job name
#SBATCH --output=output.txt    # Specify the output file
#SBATCH --error=error.txt      # Specify the error file
#SBATCH --mail-user=your.email@ubc.ca  # Email address for job notifications
#SBATCH --mail-type=ALL        # Receive email notifications for all job event
#Load necessary modules
module load gcc
module load cuda
module load apptainer

#Change to working directory
cd $SLURM_SUBMIT_DIR

#Execute the apptainer .sif file
apptainer exec --nv /arc/project/<alloc-code>/<cwl>/tensorflow/tensorflow-gpu.sif

#<commands to execute...>
```

# Running nps-screening
1. SSH to sockeye
2. Load environment
```shell
cd /arc/project/st-cfjell-1/apptainer
module load gcc
module load apptainer
apptainer shell ubuntu.sandbox
```
3. Run script to generate slurm batch script
```shell
Rscript /arc/project/st-cfjell-1/git/nps-screening/R/MS/outer-write-ms1-matches-xcms.R

```

# To-do
1. retention time
2. see RT panel Fig 2 of paper
    2. Sandrine has retention time examples, we can train to predict RT on all samples
    3. Our data has RT for each compound
3. Filter results based on predicted RT compared to measured RT
4. Look for clusters of samples that are outside predicted RT range x 
2. mass accuracy
3. Mike's Nature Machine Learning
4. See H:\Aaron_stuff-nps-rt-main.zip 


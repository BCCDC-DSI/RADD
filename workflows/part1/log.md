
# 2024-04-22

## Interactive session 

```
#### Launch interactive session
$ salloc --time=1:0:0 --mem=3G --nodes=1 --ntasks=2 --account=st-cfjell-1

#### Load Chris' apptainer
$ cd /arc/project/st-cfjell-1/apptainer
$ module load gcc
$ module load apptainer

#### Launch R Studio
$ R


### mzml_file,
# database_dir = "/arc/project/st-cfjell-1/ms_data/Data/reference/",
# output_dir = "/scratch/st-cfjell-1/output/ms_data/expedited_2023/"

source('/arc/project/st-cfjell-1/git/mass_spec/go.R')
go( "/arc/project/st-cfjell-1/ms_data/expedited_2023/mzML/2023-2649BG01.mzML"  )

```


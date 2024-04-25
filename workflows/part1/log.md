
# 2024-04-25

## Interactive session 

```
#### Launch interactive session
salloc --time=1:0:0 --mem=3G --nodes=1 --ntasks=2 --account=st-cfjell-1

#### Load Chris' apptainer
export LC_ALL=C; unset LANGUAGE
cd /arc/project/st-cfjell-1/apptainer

module load gcc
module load apptainer
apptainer shell --bind /arc/software,/arc/home,/arc/project,/scratch ubuntu.sandbox

#### Launch R Studio
R


### mzml_file,
# database_dir = "/arc/project/st-cfjell-1/ms_data/Data/reference/",
# output_dir = "/scratch/st-cfjell-1/output/ms_data/expedited_2023/"

source('/arc/project/st-cfjell-1/git/mass_spec/go.R')
go( "/arc/project/st-cfjell-1/ms_data/expedited_2023/mzML/2023-2649BG01.mzML"  )

```


## 2024-04-25

- After adding line to bind the shell, able to progress a bit:
   
  ```
  Detecting mass traces at 25 ppm ... OK
  Detecting chromatographic peaks in 31139 regions of interest ... OK: 7601 found.
  ```

- Getting error:
  ```
    # save all MS/MS spectra detected in this file
    spectra_df = data.frame(spectrum_name = names(spectra)) %>%
      separate(spectrum_name, into = c('chrom_peak', 'file', 'spectrum'),
               sep = "\\.", remove = FALSE) %>%
      dplyr::select(-file)
  
  ```

- Close inspection shows:
  ```
  > names(spectra)
  NULL
  ```






library(tidyverse)
library(magrittr)
library(xcms)
options(stringsAsFactors = FALSE)
library(argparse)

args = parser$parse_args()
print(args)

output_dir = "/scratch/st-ashapi01-1/ms_data/expedited_2023/"
database_dir = "/arc/project/st-cfjell-1/ms_data/Data/reference/"
input_dir = "/arc/project/st-cfjell-1/ms_data/expedited_2023/mzML/"


# list mzML files
if (1) {
  mzml_files = list.files(input_dir, full.names = TRUE, pattern = "mzML") 
} else {
  mzml_files = list.files(input_dir, full.names = TRUE, pattern = "mzML") %>% 
  # drop calibration, mix, QC files
  extract(!grepl('^Cal|^MeOH|^QC|^Mix', basename(.))) %>% 
  # keep only E/R 
  extract(grepl('^E|^R', basename(.)))
}

##### <--- LT ended here today 2024-05-28

# get peak-picked files
mzml_filenames = gsub("\\.mzML$", "", basename(mzml_files))
rds_files = file.path(args$output_dir, paste0(mzml_filenames, '.rds'))
stopifnot(all(file.exists(rds_files)))

# read all files
dats = map(rds_files, readRDS) %>% setNames(mzml_files )
# extract spectra
chromPeakSpectra = map(dats, 'chromPeakSpectra') %>% 
  bind_rows(.id = 'file')
# extract matches
matches_ms1 = map(dats, 'ms1_matches') %>% 
  map(~ extract(., lengths(.) > 0) %>% map('ms1') %>% bind_rows()) %>% 
  bind_rows(.id = 'file')
matches_ms2 = map(dats, 'ms1_matches') %>% 
  map(~ extract(., lengths(.) > 0) %>% map('ms2') %>% bind_rows()) %>% 
  bind_rows(.id = 'file')

# save all to RDS
output = list(chromPeakSpectra = chromPeakSpectra,
              matches = list(ms1 = matches_ms1, ms2 = matches_ms2))

# save
output_dir = dirname(args$output_file)
if (!dir.exists(output_dir))
  dir.create(output_dir, recursive = TRUE)
saveRDS(output, args$output_file)

# remove all other RDS files
if (file.exists(args$output_file))  {
  dat = readRDS(args$output_file)
  if (length(dat) == 2 & 
      all(names(dat) %in% c('chromPeakSpectra', 'matches')))
    file.remove(rds_files)
}

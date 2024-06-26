# Find chromatographic peaks with xcms and extract ones with a precursor mass
# and at least one fragment matching a compound in the NPS database.

# NPS_GIT_ROOT = Sys.getenv("NPS_GIT_ROOT")
# MS_DATA_INPUT_DIR = Sys.getenv("MS_DATA_INPUT_DIR")

# setwd(NPS_GIT_ROOT)
options(stringsAsFactors = FALSE)


library(argparse)

# parse arguments
parser = ArgumentParser(prog = 'outer-write-ms1-matches-xcms.R')
parser$add_argument('--allocation', type = 'character', default = "root")
args = parser$parse_args()

library(tidyverse)
library(magrittr)

# load grid functions
# source("R/functions/submit_job.R")
# source("R/functions/write_sh.R")
# source("R/functions/detect_system.R")

# define input directory
# input_dir = file.path(base_dir, "UDS", "converted") %>%
#   gsub("scratch", "project", .)

DATA_DIR = '/home/chris/ms_data/data/dataset1/'
REFERENCE_DIR = "/home/chris/ms_data/data/reference"
db1_filename = paste(REFERENCE_DIR, "NPS DATABASE-NOV2022.csv", sep = "/")
db2_filename = paste(REFERENCE_DIR, "THERMO DB-NOV2022.csv", sep = "/")
output_dir = "/scratch/cfjell/MS/output/dataset1/"


REFERENCE_DIR = "/arc/project/st-cfjell-1/ms_data/Data/reference/"  # library of templates to match
MS_DATA_INPUT_DIR = "/arc/project/st-cfjell-1/ms_data/expedited_2023/mzML/"
output_dir = "/scratch/st-cfjell-1/output/ms_data/expedited_2023/"



# list mzML files
mzml_files = list.files(MS_DATA_INPUT_DIR, full.names = TRUE, pattern = "mzML")
 # %>%
  # drop calibration, mix, QC files
  # extract(!grepl('^Cal|^MeOH|^QC|^Mix', basename(.))) %>%
  # # keep only E/R
  # extract(grepl('^E|^R', basename(.)))

# set up grid
grid = tidyr::crossing(
  ## CentWaveParam settings
  snthresh = 10,
  noise = 100, 
  ppm = 25, 
  peakwidth_min = 5,
  peakwidth_max = 20
)

# define output directories
# output_dir = file.path(base_dir, "xcms")

output_dir = '/scratch/st-cfjell-1/output/ms_data/rerun2024'

if (!dir.exists(output_dir))
  dir.create(output_dir)
output_dirnames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-')
})
# add into the grid
grid %<>% mutate(output_dir = file.path(output_dir, output_dirnames))

# now, cross with mzml files
grid %<>% tidyr::crossing(mzml_file = mzml_files)

# add full filenames into the grid
output_filenames = with(grid, gsub("\\.mzML$", "", basename(mzml_file))) %>% 
  paste0('.rds')
grid %<>% mutate(output_file = file.path(output_dir, output_filenames))

# now, check for which parameters are already complete
grid0 = grid %>%
  filter(!file.exists(output_file))
# remove columns that are all NA
keep = map_lgl(grid0, ~ any(!is.na(.x) & .x != 'NA'))
grid0 %<>% extract(, keep)

# write the grid that still needs to be run 
grid_file = "sh/grids/write-ms1-matches-xcms.txt"
grid_dir = dirname(grid_file)
if (!dir.exists(grid_dir))
  dir.create(grid_dir, recursive = TRUE)
write.table(grid0, grid_file, quote = FALSE, row.names = FALSE, sep = "\t")

# write the sh file dynamically
sh_file = paste0(NPS_GIT_ROOT, '/sh/write-ms1-matches-xcms.sh')
print(paste("sh_file", sh_file))
sh_dir = dirname(sh_file)
if (!dir.exists(sh_dir)) 
  dir.create(sh_dir, recursive = TRUE)

write_sh(job_name = 'write-ms1-matches-xcms',
         sh_file = sh_file,
         grid_file = grid_file,
         inner_file = 'R/MS/inner-write-ms1-matches-xcms.R',
         system = system,
         time = 24,
         mem = 16)

# finally, run the job on whatever system we're on
# submit_job(grid0, sh_file, args$allocation, system)

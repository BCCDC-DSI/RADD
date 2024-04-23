### cfjell 2023/05/30
# workflow based on tutorial for xcms
# http://bioconductor.org/packages/release/bioc/vignettes/xcms/inst/doc/xcms-lcms-ms.html

library(tictoc)
library(magrittr)
library(xcms)
options(stringsAsFactors = FALSE)
library(argparse)
library(BiocParallel)
# library(batchtools)
# register(BatchtoolsParam(workers = 7), default = TRUE)
options(MulticoreParam=MulticoreParam(workers=4))

database_dir = "/home/cfjell/data/mass_spec/data/reference/"
data_dir = "/home/cfjell/data/mass_spec/data/dataset1/"
output_dir = "/home/cfjell/data/mass_spec/output/"

# database_dir = "/scratch/cfjell/MS/data/reference"
# output_dir = "/scratch/cfjell/MS/output/dataset1/"
db1_filename = paste(database_dir, "NPS DATABASE-NOV2022.csv", sep = "/")
db2_filename = paste(database_dir, "THERMO DB-NOV2022.csv", sep = "/")



# mzml_file = 'E3390286054.mzML'
# mzml_file = paste(data_dir, mzml_file, sep = "/")
# meta = data.frame(file = mzml_file)

# read data, uses MSnbase package
### run parallel ?
mzml_files = list.files(data_dir, full.names = TRUE, pattern = "E.*mzML")
mzml_files = mzml_files[1:4]
meta = data.frame(file = mzml_files)

raw_data = readMSData(files = mzml_files,
                 pdata = new("NAnnotatedDataFrame", meta),
                 mode = "onDisk")

# peak detection on purely chromatographic data
snthresh = 10     ### using values from outer-write-ms1-matches-xcms.R
noise = 100
ppm = 25
peakwidth_min = 5
peakwidth_max = 20

cwp = CentWaveParam(snthresh = snthresh,
                    noise = noise,
                    ppm = ppm,
                    peakwidth = c(peakwidth_min, peakwidth_max))
dat = findChromPeaks(raw_data, param = cwp)
# > class(dat)
# [1] "XCMSnExp"
# attr(,"package")
# [1] "xcms"

# extract MS/MS spectra
## msLevel = 2L:
# For msLevel = 2L MS2 spectra are returned for a chromatographic peak if
# their precursor m/z is within the retention
# time and m/z range of the chromatographic peak.
## return.type character(1) defining the result type.
#  Defaults to return.type = "MSpectra" but
#  return.type = "Spectra" or return.type = "List" are preferred.
#  See below for more information.
spectra = chromPeakSpectra(dat, msLevel = 2L, return.type = "Spectra")
# spectra
# MSpectra with 2044 spectra and 1 metadata column(s):
#                     msLevel     rtime peaksCount |     peak_id
#                   <integer> <numeric>  <integer> | <character>
#   CP0015.F1.S0050         2   5.21545         14 |      CP0015
#   CP0017.F1.S0131         2  13.33396         12 |      CP0017
#   CP0289.F1.S0081         2   8.40583         18 |      CP0289

# following needs BiocManager::install("Spectra"), but fails due to libxml2 error
# spectra = chromPeakSpectra(dat, msLevel = 2L, return.type="Spectra")


db1 = read.csv(db1_filename, skip = 5)
db2 = read.csv(db2_filename, skip = 5)
  # databases = list(NPS = db1, Thermo = db2) %>%
  #   # filter to MS1/MS2 rows only
  #   map(~ {
  #     db = .x
  #     stop_at = which(db$Compound.Name == "") %>% head(1)
  #     db %<>% extract(seq_len(stop_at), )
  #     keep = map_lgl(db, ~ n_distinct(.x) > 1)
  #     db %<>% extract(, keep)
  #   })

# starting vignette
library(xcms)
dda_file = mzml_file
dda_data <- readMSData(dda_file, mode = "onDisk")

library(magrittr)

cwp <- CentWaveParam(snthresh = snthresh, noise = noise, ppm = ppm,
                     peakwidth = c(peakwidth_min, peakwidth_max))
dda_data <- findChromPeaks(dda_data, param = cwp)


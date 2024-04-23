### cfjell 2023/05/30
# workflow based on tutorial for xcms
# http://bioconductor.org/packages/release/bioc/vignettes/xcms/inst/doc/xcms-lcms-ms.html

library(tictoc)
library(magrittr)
library(xcms)
options(stringsAsFactors = FALSE)
library(argparse)


DATA_DIR = '/home/chris/ms_data/data/dataset1/'
REFERENCE_DIR = "/home/chris/ms_data/data/reference"
db1_filename = paste(REFERENCE_DIR, "NPS DATABASE-NOV2022.csv", sep = "/")
db2_filename = paste(REFERENCE_DIR, "THERMO DB-NOV2022.csv", sep = "/")
output_dir = "/scratch/cfjell/MS/output/dataset1/"


mzml_file = paste(DATA_DIR, 'R3400057402.mzML', sep='/')
meta = data.frame(file = mzml_file)

# read data, uses MSnbase package
dat = readMSData(files = mzml_file,
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
dat = findChromPeaks(dat, param = cwp)
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
spectra = chromPeakSpectra(dat, msLevel = 2L)
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



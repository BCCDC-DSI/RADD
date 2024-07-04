library(argparse)
library(tidyverse)
library(magrittr)
library(xcms)



database_file = '/arc/project/st-ashapi01-1/RADD_libraries/HRN_2023-10-01_v4_v5.csv'
#db1 = read.csv( database_file, skip = 5) 
#head(db1)

database_file = '/arc/project/st-ashapi01-1/RADD_libraries/HRN_trimmed.csv'
db1 = read.csv( database_file, skip = 5)
head(db1)
mzml_file = '/arc/project/st-cfjell-1/ms_data/expedited_2023/mzML/2024-0581BG01.mzML'

snthresh <- 10     ### using values from outer-write-ms1-matches-xcms.R
noise <- 100
ppm <- 25
peakwidth_min <- 5
peakwidth_max <- 20





# create metadata frame
meta = data.frame(file = mzml_file)

# read data
dat = readMSData(files = mzml_file, 
                 pdata = new("NAnnotatedDataFrame", meta),
                 mode = "onDisk")

# run peak detection
## can probably run with a grid of snthresh, peakwidth, ppm, noise
cwp = CentWaveParam(snthresh = snthresh,
                    noise = noise, 
                    ppm = ppm,
                    peakwidth = c(peakwidth_min, peakwidth_max))
dat = findChromPeaks(dat, param = cwp)

# extract MS/MS spectra
spectra = chromPeakSpectra(dat, msLevel = 2L)

# save all MS/MS spectra detected in this file
spectra_df = data.frame(spectrum_name = names(spectra)) %>% 
  separate(spectrum_name, into = c('chrom_peak', 'file', 'spectrum'),
           sep = "\\.", remove = FALSE) %>% 
  dplyr::select(-file)


databases = list(NPS = db1) %>% 
  # filter to MS1/MS2 rows only
  map(~ {
    db = .x
    stop_at = which(db$Compound.Name == "") %>% head(1)
    db %<>% extract(seq_len(stop_at), )
    keep = map_lgl(db, ~ n_distinct(.x) > 1)
    db %<>% extract(, keep)
  })

# function to calculate ppm boundary
calc_ppm_range = function(theor_mass, err_ppm = 10) {
  err_ppm = err_ppm[1]    # <--- 
  print( paste( 'theor_mass', theor_mass, 'err', err_ppm )
  c(
    (-err_ppm / 1e6 * theor_mass) + theor_mass,
    (err_ppm / 1e6 * theor_mass) + theor_mass
  )
}





# iterate through compounds
compounds = unique(databases$NPS$Compound.Name) %>% na.omit()

## remove one compound already in the in-house database
## compounds %<>% setdiff(c('', 'Norfluorodiazepam'))


results = map(seq_along(compounds), ~ {
  compound = compounds[[.x]]
  message("[", .x, "/", length(compounds), "] ", compound, " ...")
  
  # get parent compound info to extract candidate spectra
  compound = filter(databases$NPS, Compound.Name == compound)
  parent = filter(compound, Workflow == 'TargetPeak')
  fragments = filter(compound, Workflow == 'Fragment')
  mz_range = calc_ppm_range(parent$m.z, err_ppm = 10)

  # find spectra that match parent properties
  ms1_match = map_lgl(seq_along(spectra), ~ {    
    spectrum = spectra[[.x]]
    between(precursorMz(spectrum), mz_range[1], mz_range[2]) & 
      precursorIntensity(spectrum) >= parent$Height.Threshold[1]
  })
      
  # for the spectra that match, filter to those that contain a matching fragment
  ms2_match = map_lgl(seq_along(spectra), ~ {
    if (!ms1_match[.x]) return(FALSE)
    spectrum_df = spectra[[.x]] %>% as.data.frame()
    match = tidyr::crossing(spectrum_df, target_mz = fragments$Product.m.z) %>% 
      mutate(match = map2_lgl(mz, target_mz, ~ {
        range = calc_ppm_range(.y, err_ppm = 20)
        between(.x, range[1], range[2])
      }))
    any(match$match)
  })
    
  # abort if no matches at all
  if (!any(ms2_match)) return(list())
  
  # extract just those spectra
  spectrum_names = names(spectra)[ms2_match]
  spectrum_ms1 = map(which(ms2_match), ~ {
    spectrum = spectra[[.x]]
    data.frame(mz = precursorMz(spectrum),
               rt = rtime(spectrum),
               intens = precursorIntensity(spectrum))
  }) %>% 
    setNames(spectrum_names) %>% 
    bind_rows(.id = 'spectrum')
  spectrum_ms2 = map(which(ms2_match), ~ as.data.frame(spectra[[.x]])) %>% 
    setNames(spectrum_names) %>% 
    bind_rows(.id = 'spectrum')
  output = list(ms1 = spectrum_ms1, ms2 = spectrum_ms2) %>% 
    map(~ cbind(parent, .x))
}) %>% 
  setNames(compounds)






# add chromPeakSpectra results
output = list(chromPeakSpectra = spectra_df,
              ms1_matches = results)

# save
output_dir = dirname(args$output_file)
if (!dir.exists(output_dir))
  dir.create(output_dir, recursive = TRUE)
saveRDS(output, args$output_file)

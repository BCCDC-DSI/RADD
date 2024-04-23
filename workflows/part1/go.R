# library(tictoc)
library(tidyverse)
library(magrittr)
library(xcms)
options(stringsAsFactors = FALSE)
library(argparse)

go_all <- function(data_dir = "/arc/project/st-cfjell-1/ms_data/expedited_2023/mzML/") {
    mzml_files = list.files(data_dir, full.names = TRUE, pattern = "mzML")
    N = length(mzml_files)
    i = 0
    for (mzml_file in mzml_files) {
        i = i + 1
        print(paste("mzml_file:", mzml_file, "(", i, "/", N, ")", sep = ""))
        tictoc::tic()
        go(mzml_file)
        tictoc::toc()
    }
}

gotest <- function(mzml_file) {

  data_dir = "/scratch/cfjell/MS/data/dataset1"
  database_dir = "/arc/project/st-cfjell-1/ms_data/Data/reference/"
  mzml_file = paste(data_dir, mzml_file, sep = "/")
  print(paste("mzml_file:", mzml_file))
}


go <- function(mzml_file,
              database_dir = "/arc/project/st-cfjell-1/ms_data/Data/reference/",
              output_dir = "/scratch/st-cfjell-1/output/ms_data/expedited_2023/"
               ) {

  snthresh = 10     ### using values from outer-write-ms1-matches-xcms.R
  noise = 100
  ppm = 25
  peakwidth_min = 5
  peakwidth_max = 20

  db1_filename = paste(database_dir, "NPS DATABASE-NOV2022.csv", sep = "/")
  db2_filename = paste(database_dir, "THERMO DB-NOV2022.csv", sep = "/")


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

  # now, read databases
  db1 = read.csv(db1_filename, skip = 5)
  db2 = read.csv(db2_filename, skip = 5)
  databases = list(NPS = db1, Thermo = db2) %>%
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
    ## error (in ppm) = (measured - theoretical) / theoretical * 1e6
    c(
      (-err_ppm / 1e6 * theor_mass) + theor_mass,
      (err_ppm / 1e6 * theor_mass) + theor_mass
    )
  }

  # iterate through compounds
  compounds = unique(databases$NPS$Compound.Name) %>% na.omit()
  ## remove one compound already in the Thermo database
  compounds %<>% setdiff(c('', 'Norfluorodiazepam'))
  results = map(seq_along(compounds), ~ {
    compound = compounds[[.x]]
    message("[", .x, "/", length(compounds), "] ", compound, " ...")

    # get parent compound info to extract candidate spectra
    compound = filter(databases$NPS, Compound.Name == compound)
    parent = filter(compound, Workflow == 'TargetPeak')
    fragments = filter(compound, Workflow == 'Fragment')
    mz_range = calc_ppm_range(parent$m.z, err_ppm = 10)
    ## do not filter based on RT for now
    # rt_range = with(parent, c(Retention.Time - Retention.Time.Window,
    #                           Retention.Time + Retention.Time.Window))

    # find spectra that match parent properties
    ms1_match = map_lgl(seq_along(spectra), ~ {
      spectrum = spectra[[.x]]
      between(precursorMz(spectrum), mz_range[1], mz_range[2]) &
        ## between(rtime(spectrum), rt_range[1], rt_range[2]) &
        precursorIntensity(spectrum) >= parent$Height.Threshold
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
  output_file = paste(strsplit(x=basename(mzml_file), split=".", fixed=TRUE)[[1]][1], "rds", sep=".")
  output_file = paste(output_dir, output_file, sep="/")
  print(paste0("writing to ", output_file))
  saveRDS(output, output_file)
}

parse_outputs <- function(output_dir) {
  rds_files = list.files(output_dir, pattern = "*.rds", full.names = TRUE)
  compounds = list()
  for(rds_file in rds_files){
    output = readRDS(rds_file)
    output_obj = output[[2]]
    found_compounds = c()
    for(i in 1:length(output_obj)) {
      x = output_obj[[i]]
      if(length(x) > 0){

        if(names(output_obj)[i] != "Eutylone"){
          print(names(output_obj)[i])
          return(1)
        }
        found_compounds = c(found_compounds, names(output_obj)[i])
      }
    }
    compounds[[basename(rds_file)]] = paste(sort(found_compounds), collapse=', ')
   }
    rv_df = data.frame(sample = names(compounds), found = unlist(compounds))
}

# take argument from command line of "filename" and pass it to go()
args = commandArgs(trailingOnly=TRUE)
if (length(args)==1) {
  print(paste0("Run go() with ", args[1]))
  go(args[1])
}


# Add path so Chris' aptainer will be used
.libPaths( c( '/arc/project/st-ashapi01-1/RADD/library_final/R/x86_64-pc-linux-gnu-library/4.3', .libPaths() ))

library(tidyverse)
library(magrittr)
library(xcms)
options(stringsAsFactors = FALSE)
library(argparse)

args=commandArgs(trailingOnly=TRUE)

#if (exists('args')==FALSE){ args = commandArgs(trailingOnly=TRUE) }

if (length(args)>1)
{
  print(paste("Run inner.R with arguments: folder containing mzML:", args[1], '\nLibrary path:', args[2], '\nOutput folder:', args[3], '\nAnalyzing batch #',args[4], 'of', args[5], '\n\n\n' ))
  inp_dir = args[1]
  db_filename = args[2]
  out_dir = args[3]
  NUM = as.numeric( args[4] )
  NFOLDS = as.numeric( args[5] )
  
} else {
  inp_dir = '/arc/project/st-cfjell-1/ms_data/expedited_2023/mzML/'
  out_dir = "/scratch/st-ashapi01-1/debug/"
  db_filename = '/arc/project/st-ashapi01-1/RADD_libraries/HRN_2023-10-01_v4_v5.csv'
}

## ================== Generate file list

if (exists('DEBUG')==FALSE){ DEBUG=0 }
if (DEBUG)
{
  database_dir = "/arc/project/st-cfjell-1/ms_data/Data/reference/"
  db_filename = paste( database_dir, "NPS DATABASE-NOV2022.csv", sep = "/")
  FILES = list( '/arc/project/st-cfjell-1/ms_data/expedited_2023/mzML/2024-0581BG01.mzML' )
} else {
  FILES = list.files( inp_dir , '*mzML')
  print(paste( 'Found', length(FILES), 'in the input folder...', inp_dir ))
  FILES = FILES[ seq(NUM,length(FILES), NFOLDS) ]
  print(paste('Working in batch mode; current batch has', length(FILES), 'files to process...' ))
}


## ================== Database

db1 = read.csv(db_filename, skip = 5)
print(sprintf('Reading database %s with %d rows', db_filename, nrow(db1) ))


## ================== output folder

subfolder = paste(strsplit(x=basename(db_filename), split=".", fixed=TRUE)[[1]][1])
out_dir = sprintf( '%s/%s/', out_dir, subfolder )
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
  cat("Output directory created:", out_dir, "\n")
} else {
  cat("Output directory already exists:", out_dir, "\n")
}




calc <- function( mzml_file, db1, output_file )
{
  snthresh = 10     ### values from outer-write-ms1-matches-xcms.R
  noise = 100
  ppm = 25
  peakwidth_min = 5
  peakwidth_max = 2
  
  meta = data.frame(file = mzml_file)
  message( sprintf( 'Meta df: %d x %d', nrow(meta), ncol(meta) )); flush.console()

  # read data
  dat0 = readMSData(files = mzml_file,
                   pdata = new("NAnnotatedDataFrame", meta),
                   mode = "onDisk")
  
  message( sprintf("mzML read %d row(s) of data; database has %d rows and %d cols", nrow(dat0), nrow(db1), ncol(db1) )); flush.console()
  
  # run peak detection
  ## can probably run with a grid of snthresh, peakwidth, ppm, noise
  cwp = CentWaveParam(snthresh = snthresh,
                      noise = noise,
                      ppm = ppm,
                      peakwidth = c(peakwidth_min, peakwidth_max))

  dat = findChromPeaks(dat0, param = cwp)
  
  message( sprintf("Step 1 (ChromPeak detection) completed; %d x %d", nrow(dat), ncol(dat) )); flush.console()

  # extract MS/MS spectra
  spectra = chromPeakSpectra(dat, msLevel = 2L)
  
  # save all MS/MS spectra detected in this file
  spectra_df = data.frame(spectrum_name = names(spectra)) %>%
    separate(spectrum_name, into = c('chrom_peak', 'file', 'spectrum'),
             sep = "\\.", remove = FALSE) %>%
    dplyr::select(-file)
   
  message( sprintf("Step 2+3: spectra extracted and compiled to df as, %d x %d", nrow(spectra_df), ncol(spectra_df) )); flush.console()
  
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
    ## error (in ppm) = (measured - theoretical) / theoretical * 1e6
    c(
      (-err_ppm / 1e6 * theor_mass) + theor_mass,
      (err_ppm / 1e6 * theor_mass) + theor_mass
    )
  }
  
  # iterate through compounds
  compounds = unique(databases$NPS$Compound.Name) %>% na.omit()
  ## remove one compound already in the Thermo database
  # compounds %<>% setdiff(c('', 'Norfluorodiazepam'))
  
  message( "Step 5: matching procedure begins" ); flush.console()
  
  results = map(seq_along(compounds), ~ {
    compound = compounds[[.x]]
    message("[", .x, "/", length(compounds), "] ", compound, " ..."); flush.console()
    
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
      # print( paste0( 'precursor:', dim( precursorMz(spectrum) )) )
      between(precursorMz(spectrum)[1], mz_range[1], mz_range[2]) &
        ## between(rtime(spectrum), rt_range[1], rt_range[2]) &
        precursorIntensity(spectrum) >= parent$Height.Threshold[1] #<-- Bug-fix 3: "[1]" so only the first value is piped
    })
    
    
    # for the spectra that match, filter to those that contain a matching fragment
    ms2_match = map_lgl(seq_along(spectra), ~ {
      if (!ms1_match[.x]) return(FALSE)
      spectrum_df = spectra[[.x]] %>% as.data.frame()
      match = tidyr::crossing(spectrum_df, target_mz = fragments$Product.m.z) %>%
        mutate(match = map2_lgl(mz, target_mz, ~ {
          range = calc_ppm_range( as.numeric(.y), err_ppm = 20)
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
    setNames(compounds) # end of assignment to "results" variable
  
  
  # add chromPeakSpectra results
  output = list(chromPeakSpectra = spectra_df, ms1_matches = results)
  # save
  message( "writing to ", output_file, Sys.time() ); flush.console()
  saveRDS(output, output_file)
}



i=-1
for (filename in FILES)
{  
  i=i+1
  if (1)      # (( i %/% num) == 0)
  {
    mzml_file = paste0( inp_dir, filename)
    print( paste( 'Processing', mzml_file ) )
    
    prefix = paste(strsplit(x=basename(mzml_file), split=".", fixed=TRUE)[[1]][1], "rds", sep=".")
    output_file = paste(out_dir, prefix, sep="/")
    
    if (file.exists( output_file ))
    {
      print( paste( output_file, 'already exists'))
      
    } else {      
      print(sprintf( 'Will output to %s [%d of %d files]', output_file, i, length(FILES) ))        
      calc( mzml_file, db1, output_file )


      
    } # if file exists, skip so won't repeat calculations    
  } # if selected to be processed in current subjob
} # end of loop



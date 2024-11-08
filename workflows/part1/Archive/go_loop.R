
# Add libraries so the apptainer with old XCM version is first read and subsequently used (instead of the latest XCM)
.libPaths( c( '/arc/project/st-ashapi01-1/RADD/library_final/R/x86_64-pc-linux-gnu-library/4.3', .libPaths() ))

library(tidyverse)
library(magrittr)
library(xcms)
options(stringsAsFactors = FALSE)
library(argparse)

# DEBUG=1; source('/arc/project/st-ashapi01-1/RADD/workflow1/inner2.R') # works as of 2024-07-05
# DEBUG=0; source('/arc/project/st-ashapi01-1/RADD/workflow1/inner2.R') # works as of 2024-07-05
# source('/arc/project/st-ashapi01-1/RADD/workflow1/inner3.R') # works as of 2024-07-05
#
# mzml_file = '/arc/project/st-cfjell-1/ms_data/expedited_2023/mzML/2024-0581BG01.mzML'
# database_dir = "/arc/project/st-cfjell-1/ms_data/Data/reference/"
# /arc/project/st-ashapi01-1/RADD_libraries/HRN_2023-10-01_v4_v5.csv

args=commandArgs(trailingOnly=TRUE)

if (length(args)>1) # Excecuted in an offline SLURM job
{
  print(paste("Run inner.R with arguments: folder containing mzML:", args[1], '\nLibrary path:', args[2], '\nOutput folder:', args[3], '\nAnalyzing batch #',args[4], 'of', args[5], '\n\n\n' ))
  inp_dir = args[1]
  db_filename = args[2]
  out_dir = args[3]
  NUM = as.numeric( args[4] )
  NFOLDS = as.numeric( args[5] )  
  iDEBUG = as.numeric( args[6] ) 

} else { # Executed interactively 
  if ( exists('inp_dir') == FALSE ) {
	  inp_dir = '/arc/project/st-cfjell-1/ms_data/expedited_2023/mzML/'  
	  inp_dir = '/arc/project/st-cfjell-1/ms_data/expedited_2024/'
  }
  if ( exists('db_filename') == FALSE ) {  
	  db_filename = '/arc/project/st-ashapi01-1/RADD_libraries/HRN_2023-10-01_v4_v5.csv';
	  db_filename = '/arc/project/st-ashapi01-1/RADD_libraries/NPS_DB-240705.csv' 
  }
  if ( exists( 'out_dir') == FALSE) { out_dir = '/scratch/st-ashapi01-1/expedited_2024/' }
  if ( exists('NFOLDS') == FALSE) { NFOLDS = 1; }
  # /scratch/st-cfjell-1/inputs/ms_data/
}



# =================== database read 
db1 = read.csv(db_filename, skip = 5) # colClasses=c('numeric','numeric') )

print( 'Inspect the database closely; any numeric values enclosed with quotes??? sign of problem later!')

if (1){
d=3; db1[,d] <- as.numeric(db1[,d])
d=7; db1[,d] <- as.numeric(db1[,d])
d=21; db1[,d] <- as.numeric(db1[,d])
d=22; db1[,d] <- as.numeric(db1[,d])
}

print(  t(head(db1,1)) )
print( '\n\n' )
cat( '\n\nRunning copy under:\n\t/arc/project/st-ashapi01-1/git/ywtang/RADD/workflows/part1/go_loop.R\n\n\n' )



d=7; db1[,d] = as.numeric(db1[,d])
d=5; db1[,d] = as.numeric(db1[,d])

# ================== output subfolder path
subfolder = paste(strsplit(x=basename(db_filename), split=".", fixed=TRUE)[[1]][1])
out_dir = sprintf( '%s/%s/', out_dir, subfolder )
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
  cat("Output directory created:", out_dir, "\n")
} else {
  cat("Output directory already exists:", out_dir, "\n")
}



# =================== compile list of files to process for this batch
FILES = list.files( inp_dir , '*mzML')
print(paste( 'Found', length(FILES), 'in the input folder...', inp_dir ))

if ( NFOLDS > 1) {
	FILES = FILES[ seq(NUM,length(FILES), NFOLDS) ]
	print(paste('Working in batch mode; current batch has', length(FILES), 'files to process...' ))
}


cat('Will not check for Compound.Name when defining database variable 2024-08-02')

i=-1
for (filename in FILES)
{  
  i=i+1
  mzml_file = sprintf( '%s/%s', inp_dir, filename)
  print( paste( 'Processing', mzml_file ) )
  
  prefix = paste(strsplit(x=basename(mzml_file), split=".", fixed=TRUE)[[1]][1], "rds", sep=".")
  output_file = paste(out_dir, prefix, sep="/")
  
  if (file.exists( output_file ))
  {
    print(paste( output_file, 'already exists; skip to next file'))    
  } 
  else 
  {
    print(sprintf( 'Will output to %s [%d of %d files]', output_file, i, length(FILES) ))        
    
    snthresh = 10     ### using values from outer-write-ms1-matches-xcms.R
    noise = 100
    ppm = 25
    peakwidth_min = 5
    peakwidth_max = 20
    
    # # read data + metadata dfs
    meta = data.frame(file = mzml_file)  
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
        keep = map_lgl(db, ~ n_distinct(.x) > 1)
        db %<>% extract(, keep)
      })

  
    # function to calculate ppm boundary
    calc_ppm_range = function(theor_mass, err_ppm = 10.0) {
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
    
   
    results = map(seq_along(compounds), ~ {
      compound = compounds[[.x]]
      #if (0){ message("[", .x, "/", length(compounds), "] ", compound, " ...") }
  
      # get parent compound info to extract candidate spectra
      compound = filter(databases$NPS, Compound.Name == compound)
      parent = filter(compound, Workflow == 'TargetPeak')
      #if (0){ message( 'parent m.z: ', parent$m.z, ' row:', .x ) }
      fragments = filter(compound, Workflow == 'Fragment')
      mz_range = calc_ppm_range( as.numeric(parent$m.z), err_ppm = 10)
      ## do not filter based on RT for now
      rt_range = with(parent, c(Retention.Time - Retention.Time.Window,
                                Retention.Time + Retention.Time.Window))
  
      # find spectra that match parent properties
      ms1_match = map_lgl(seq_along(spectra), ~ {
        spectrum = spectra[[.x]]
	if (0) { message('Threshold', parent$Height.Threshold, 'at row:', .x ) }
        # print( paste0( 'precursor:', dim( precursorMz(spectrum) )) )
        between(precursorMz(spectrum)[1], mz_range[1], mz_range[2]) &
        between(rtime(spectrum), rt_range[1], rt_range[2]) &
          precursorIntensity(spectrum) >= parent$Height.Threshold[1]   # <---- need to change still? 
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
      setNames(compounds) 
  
    # add chromPeakSpectra results
    output = list(chromPeakSpectra = spectra_df, ms1_matches = results)  

    # save
    output_file = paste(strsplit(x=basename(mzml_file), split=".", fixed=TRUE)[[1]][1], "rds", sep=".")
    output_file = paste(out_dir, output_file, sep="/")
    saveRDS(output, output_file)

    print( paste("writing to", output_file, ' | Batch', NUM, 'of', NFOLDS, '\n\n' ))

    # Sys.time() ))  # seems to take extra 2 seconds to call system's time
    
    
  } # Rds does not exist 
} #  end of loop 

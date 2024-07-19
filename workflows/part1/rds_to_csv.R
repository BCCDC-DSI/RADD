




install.packages( 'pacman', repos = "http://cran.us.r-project.org" )
library('pacman')
p_load('argparse')


args = commandArgs(trailingOnly=TRUE)

if (args[1] == 1) 
{
	folder = 'data_highresnps' 
} else {
	folder = 'data_2024nps-db'
}

files = list.files( paste0('/scratch/st-ashapi01-1/rds_files/', folder), pattern = "*.rds", full.names = TRUE )


for (file in files)
{
# paste0('.')

dat <- readRDS(file)

f1 = list()
f2 = list()

c1 = list()
c2 = list()
c3 = list()
c4 = list()
c5 = list()
c6 = list()
c7 = list()

b1 = list()
b2 = list()
b3 = list()
b4 = list()
b5 = list()

pref = tools::file_path_sans_ext(basename(file))

for (n in names(dat$ms1_matches))
{
  #cat('\n', n)
  
  if ( length( dat$ms1_matches[[n]] )==2 )
  {
    r1 = nrow(dat$ms1_matches[[n]]$ms1)
    r2 = nrow(dat$ms1_matches[[n]]$ms2)
    
    b1=c(b1, dat$ms1_matches[[n]]$ms1$Compound.Name )
    b2=c(b2, dat$ms1_matches[[n]]$ms1$m.z  )# varying
    b3=c(b3, dat$ms1_matches[[n]]$ms1$mz   ) # varying
    b4=c(b4, dat$ms1_matches[[n]]$ms1$rt ) # varying
    b5=c(b5, dat$ms1_matches[[n]]$ms1$intens ) # varying
    
    c1 = c(c1, dat$ms1_matches[[n]]$ms2$Compound.Name)
    c2 = c(c2, dat$ms1_matches[[n]]$ms2$m.z ) # constant
    c3 = c(c3, dat$ms1_matches[[n]]$ms2$mz )  # varying
    c4 = c(c4, dat$ms1_matches[[n]]$ms2$i )
      
    ms2= dat$ms1_matches[[n]]$ms2
    ms1 = dat$ms1_matches[[n]]$ms1

    c5 = c(c5, dat$ms1_matches[[n]]$ms2$spectrum )

    rt = dat$ms1_matches[[n]]$ms1$rt
    inte = dat$ms1_matches[[n]]$ms1$intens
    
    #c6 = c(c6, rep(1, r2)*rt )
    #c7 = c(c7, rep(1, r2)*inte )
    
    f1 = c(f1, replicate( r1, pref ))
    f2 = c(f2, replicate( r2, pref ))
    
    if (r1>3)
    {
      sample1=ms1
      sample2=ms2
    }
  } # end of if-else
} # end of list of compounds

if (length(b1)>0)
{
	df1=cbind( t(data.frame(f1)), t(data.frame(b1)), t( data.frame(b2)),t( data.frame(b3)), t( data.frame(b4)) )
	colnames(df1) = c( 'filename', 'compound_name', 'rt', 'i', 'intens')

	df2=cbind( t(data.frame(f2)), t(data.frame(c1)), t( data.frame(c5)), t( data.frame(c2)), t(data.frame(c3)), t( data.frame(c4)) )
	colnames(df2) = c('filename','compound_name', 'spectrum', 'mz', 'm.z', 'i')

	#df2=cbind( t(data.frame(ff)), t(data.frame(c1)), t( data.frame(c5)), t( data.frame(c2)), t(data.frame(c3)), t( data.frame(c4)), t( data.frame(c6)), t( data.frame(c7))  )
        #colnames(df2) = c('filename','compound_name', 'spectrum', 'mz', 'm.z', 'i', 'rt', 'intens')

	print(cat( 'ms1:', dim(df1), 'ms2:', dim(df2)) )

	out1 = paste0('/scratch/st-ashapi01-1/rds_files/', folder, '/ms1/')
	out2 = paste0('/scratch/st-ashapi01-1/rds_files/', folder, '/ms2/')

	write.csv( df1, paste0( out1, pref, '_ms1.csv'), row.names=F )
	write.csv( df2, paste0( out2, pref, '_ms2.csv'), row.names=F )
}
else
{
	print( cat(pref, 'has no matches'))
}

}



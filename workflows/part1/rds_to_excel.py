
# import rds
# from rds import RDSDict
# file = '/scratch/st-ashapi01-1/RADD/2024-05-28/2023-0001BG01.rds'
# myDict = RDSDict( file, 'match1' )

import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rds2py import read_rds
from glob import glob

PWD = os.path.abspath(os.getcwd())

print( len(sys.argv), 'arguments' )  # ipython
if len(sys.argv)<2:
  print( '\n:Usage:\n')
  print( 'python rds_to_excel.py /scratch/st-ashapi01-1/rds_files/data_highresnps/ \n')
  print( 'python rds_to_excel.py /scratch/st-ashapi01-1/rds_files/data_2024nps-db/')
  exit()   
else:
  folder = sys.argv[1]

print( folder , '\n')

os.chdir(folder)
filenames=glob( '*.rds')
print( f'processing {len(filenames)} files')
input('Ready to proceed? [Press ENTER if yes]')

DEBUG = 0

FILE, NA, INTENS, I, SPEC, MS1, MS2, RT = [],[],[],[],[],[],[],[] 
for f,filename in enumerate( filenames ):
  dat = read_rds(os.path.join( folder, filename))
  for indx in range(100):
    # D = dat['data'][0] # contains 'CP0299.F1.S0177'
    d1 = 1
    D = dat['data'][d1]['data'][indx]['data']
    d2 = 1
    if len( D ) > 0:
        A = D[d2]['data']
        na = A[0]['data'][0]                 
        try:           
          spectrum = D[0]['data'][19]['data'][0]
          ms1 = D[0]['data'][20]['data'][0]
          rt = D[0]['data'][21]['data'][0]                  
          ms2 = list(A[ 20 ]['data'])
          i = list(A[ 21 ]['data'])
        except:
          spectrum = D[0]['data'][18]['data'][0]
          ms1 = D[0]['data'][19]['data'][0]
          rt = D[0]['data'][20]['data'][0]
          intens = D[0]['data'][21]['data'][0]
          
          ms2 = list(A[ 19 ]['data'])
          i = list( A[ 20 ]['data'])                    
        if DEBUG:
          print( 'rt:\n', rt  )
          print( na, '\n', 'ms1:\n',  ms1 )
          print( 'ms2:\n',  ms2 )
          print( 'intensity:\n', i )
        else:
          print(end='.', flush=True )
        
        n = len(ms2)
        FILE += [filename] * n
        NA += [na] * n
        SPEC += [spectrum] * n
        INTENS += [intens] * n
        RT += [rt] * n
        MS1 += ms1
        MS2 += ms2
        I += i

Big = pd.DataFrame( dict( filename=FILE, compound_name = NA, spectrum=SPEC, intens=INTENS, retention_time=RT, ms1=MS1, ms2=MS2, i=I) )
outpref = os.path.dirname(folder).split('/')[-1]
print( 'Now writing to', outpref )

Big.to_excel( f'{PWD}/{outpref}_data.xlsx' )
Big.to_csv( f'{PWD}/{outpref}_data.csv' )




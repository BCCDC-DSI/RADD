
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
  folder = '/scratch/st-ashapi01-1/rds_files/data_2024nps-db/'
  # folder = '/scratch/st-ashapi01-1/rds_files/data_highresnps/'
else:
  folder = sys.argv[1]

print( folder , '\n')
os.chdir(folder)
filenames=glob( '*.rds')
print( f'processing {len(filenames)} files')
input('Ready to proceed? [Press ENTER if yes]')

DEBUG = 0


FILE, NA, INT, MS1, MS2, RT = [],[],[],[], [], []
for f,filename in enumerate( filenames ):
  dat = read_rds(os.path.join( '/scratch/st-ashapi01-1/rds_files/data_2024nps-db/', filename))
  for indx in range(100):
    # D = dat['data'][0] # contains 'CP0299.F1.S0177'
    d1 = 1
    D = dat['data'][d1]['data'][indx]['data']
    d2 = 1
    if len( D ) > 0:
        A = D[d2]['data']
        na = A[0]['data'][0]
        rt = dat['data'][1]['data'][indx]['data'][0]['data'][21]['data'][0]

        ms1, ms2, i = list(A[ 6 ]['data']), list(A[ 20 ]['data']), list(A[ 21 ]['data'])

        if DEBUG:
          print( 'rt:\n', rt  )
          print( na, '\n', 'ms1:\n',  ms1 )
          print( 'ms2:\n',  ms2 )
          print( 'intensity:\n', i )
        else:
          print(end='.')
        n = len(ms1)
        FILE += [filename] * n
        NA += [na] * n
        RT += [rt] * n
        MS1 += ms1
        MS2 += ms2
        INT += i

Big = pd.DataFrame( dict( filename=FILE, compound_name = NA, retention_time=RT, ms1=MS1, ms2=MS2, i=INT) )
outpref = os.path.dirname(folder).split('/')[-1]
print( 'Now writing to', outpref )

Big.to_excel( f'{PWD}/{outpref}_data.xlsx' )
Big.to_csv( f'{PWD}/{outpref}_data.csv' )




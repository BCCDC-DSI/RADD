# Meta info: Script below is saved on sockeye at this location: /arc/project/st-ashapi01-1/git/ywtang/RADD/workflows/part4/create_dataset.py
#
# How to run:
#
# 1) Issue interactive job to run python with multiprocessing:
#    salloc --time=10:0:0 --mem=40G --nodes=1 --ntasks=48 --account=st-ashapi01-1
#
# 2) Copy-paste:
#    conda activate chemenv
#    python /arc/project/st-ashapi01-1/git/ywtang/RADD/workflows/part4/create_dataset.py
#

import multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm

pool = multiprocessing.Pool()
ncores = pool._processes

all_files = pd.read_csv('/arc/project/st-cfjell-1/ms_data/mzmL_files/file_list.csv') # file_name
pos_samps = pd.read_excel('/arc/project/st-ashapi01-1/git/afraz96/RADD/workflows/part4/Data/tbltest.xlsx', engine = 'openpyxl' )
pos_samps = pos_samps.merge( all_files, how='left', on='Case_PTC_No' )
pos_samps['filename_pref']= pos_samps.file_name.str.strip('.mzML')
pos_samps.set_index( 'filename_pref', inplace= True )

pos_samps = pos_samps[pos_samps['file_name'].notna()] 
print( 'After removing detected samples without mzML, size of pos sample is:', pos_samps.shape )

def join(S=1):
  if S==1:
    f=['filename','compound_name', 'spectrum', 'Retention.Time', 'm.z', 'mz', 'rt', 'intens' ]
  else:
    f=['filename','compound_name', 'spectrum', 'm.z', 'mz', 'i']    
    
  for d in [2020,2021,2022,2023,2024]:
    data_dir = f'/scratch/st-ashapi01-1/expedited_{d}/combined_db_20240801/'  
    subcohort=pd.read_csv( data_dir + f'combined_ms{S}.txt')[ f ]
    q=np.where( ~np.isnan( subcohort['m.z'] ) )[0]    
    print(f'Joining ms{S} (precursor ion) results from mzML acquired in', d, )  
    if d == 2020:
      new_df = subcohort
    else:
      new_df = pd.concat( [new_df, subcohort] )
  
  new_df.set_index( 'filename', inplace=True )
  return new_df
# new_df2=join(S=2)
new_df=join(S=1)
 
# Assign class to samples
#
new_df[ 'class_label' ] = 0 
def assign_class(new_df, i):
  inds = np.intersect1d( new_df.index, i )
  new_df.loc[ inds[0] , ['class_label'] ] = 1       
  return new_df

if 0:
  with Pool(ncores) as mp_pool:
      for i in tqdm(pos_samps.index[:ncores]):
          mp_pool.apply_async( assign_class, (i,))
      mp_pool.close()
      mp_pool.join()

for j,i in enumerate( tqdm(pos_samps.index)):  
  inds = np.intersect1d( new_df.index, i )
  try:
    new_df.loc[ inds[0], ['class_label'] ] = 1       
    if (j % 1000)==0:
      new_df.to_csv( '/arc/project/st-ashapi01-1/git/ywtang/RADD/workflows/part4/dataset.csv' )
      print( np.sum(new_df.class_label), 'pos samples' )
      print( np.sum(new_df.class_label==0), 'neg samples' )
  except:
    pass



# Add year of data collection
# 
new_df['year'] = new_df.index
new_df['year'] = new_df['year'].str[:4]
new_df['year'] = new_df['year'].astype(np.uint16)



dev_set  = new_df[ new_df['year'] < 2023 ]
test_set = new_df[ new_df['year'] == 2023 ]

# Prelim EDA
# 
# Add new column for classifcation label
neg_samples = new_df[ new_df['class_label'] == 0 ]
pos_samples = new_df[ new_df['class_label'] == 1 ] 

def show_stats(df,st):
  for field in ['inten', 'mz', 'm.z', 'rt' ]:
    for d in [2024,2023,2022,2021,2020]:
      inds = np.where( ~np.isnan( df[f'{field}_{d}']))[0]
      try:
        n=np.sum(~np.isnan( df[f'{field}_{d}']))
      except:
        n=np.sum(~np.isnan( df[f'{field}']))
      print( f'Subcohort {d} has {n} {st} samples that contain nonempty {field}' )
  
show_stats(neg_samples,'negative')
show_stats(pos_samples,'positive')
 

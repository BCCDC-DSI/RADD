#!/bin/bash

##### Max size of a job array is <1000 ( MaxArraySize).
#  ***Using 500 below***
#   (base) [cfjell@login01 scripts]$ scontrol show config | grep -E 'MaxArraySize|MaxJobCount'
#   MaxArraySize            = 1000
#   MaxJobCount             = 50000
# Split inputs into batches using symlinks under another directory, in batches of 1000
input_mzML_dir="/arc/project/st-cfjell-1/ms_data/expedited_2023/mzML/"
symlinked_mzML_dir="/scratch/st-cfjell-1/inputs/ms_data/expedited_2023"
files_per_dir=500
file_counter=0
dir_counter=0

mkdir -p "${symlinked_mzML_dir}/${dir_counter}"

for file in "${input_mzML_dir}"/*mzML; do
    if (( file_counter == files_per_dir )); then
        # Reset file_counter and increment dir_counter
        file_counter=0
        ((++dir_counter))
        mkdir -p "${symlinked_mzML_dir}/${dir_counter}"
    fi

    ln -sf "${file}" "${symlinked_mzML_dir}/${dir_counter}"
    ((++file_counter))
done


for subdir in $(ls ${symlinked_mzML_dir}); do
  in_dir="${symlinked_mzML_dir}/${subdir}"
  N=$(ls -1 $in_dir/*mzML | wc -l)
  echo "sbatch --array=1-$N slurm_batch.sh $in_dir"
  # sbatch --array=$start-$end slurm_batch.sh $in_dir
done

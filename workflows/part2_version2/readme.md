
# SMILES2Vec ML pipeline

- Uses the Smiles2Vec repo available [here](https://github.com/Abdulk084/Smiles2vec)
- Standard ML regression pipeline battery (to see models used please review `config/config.yaml`)

## Setup

If you are using Anaconda, we recommend creating a blank environment with the following command:

```
conda create --name raddpt2 python=3.9
```

When this <blank> environment is ready you can activate it through:

```
conda activate raddpt2
```

Then your terminal should have something like:

```
(raddpt2) User> 
```

You should be equipped to install all the dependencies under the `requirements.txt` file:

```
pip install -r requirements.txt
```


## Usage

The ML pipeline is setup in 3 stages:

1) Create the Data used
2) Train the models
3) Analyse the results

It is important to understand how each of these sections works:

### Data creation

We make use of the `create_dataset.py` file. To use this file you will need to modify the config file found under: `workflows/part2_version2/config/config.yaml` (also available [here](https://github.com/BCCDC-DSI/RADD/blob/pt2/workflows/part2_version2/config/config.yaml))

Then to create the dataset used in the analysis:

```
python create_dataset.py
```

Note: the output is sent to the folder specified in the config file.


### Train the models

To do this step we have created a shell script:

```
./smiles_job_submitter.sh
```
To change the datasets used modify the file paths to `OUTPUT_PATH` and `DATA_PATH`. Note that this step will not work unless the dataset is created correctly in the previous section.

### Analyse the results

There are various items in this repo that can be used for analysis. We recommend using the python script: `generate_model_stats.py` - this can analyse how one database can predict on another. To assist in automating this on Sockeye we have created a shell script:

```
./create_error_tables.sh
```

It is important to mention that the `TEST_DATA_PATH` denotes the data that the `DATA_PATH` models predict on. Unfortunately, due to the current structure of the pipeline, you need to manually also modify the `config.yaml` file and in particular the following params:

```
model_index : 'Compound' # Sample ID ['Name', 'Compound']
test_model_index : 'Compound'
model_y : 'Retention Time (min)'  #['PTC Confirmed RT', 'Retention Time (min)']
test_model_y : 'PTC Confirmed RT'
model_X : 'SMILES'
```

You will need to switch them around depending on which way you are analysing. The example above is using X500R on the BCCDC dataset.


Alternatively, under `Notebooks/` there is an easy to follow guide which can leverage any of the trained models and predict on the `High Res` dataset. This notebook can be easily expanded as the functions are reasonably dynamic for the `SMILES2Vec` approach. 


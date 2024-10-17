# COVID Reinfection ML Pipeline

This repository houses the software produced for the COVID reinfection pipeline. There is some setup required and then some points surrounding usage. Please read this README to run the Machine Learning Pipeline, if there are issues please raise an issue on this GitHub repository or send an email to afraz.khan@bccdc.ca

## Setup

This repository comes equipped with an anaconda virtual environment to use across various systems. Citing Mike Irvine's setup guide as a good reference for these READMEs. 

In project directory run the following:
```bash
conda env create -f environment.yml
```
This creates an environment named `mlenv` (note you can change the environment name by editing the `environment.yml` file). To activate the virtual environment run:

```bash
conda activate mlenv
```

Note that in osx you may instead need to run:
```bash
source activate mlenv
```

## Usage

To run the Machine Learning Pipeline you must type the following command (provided you have installed the required software):

```bash
python train_test.py -d <dataset> -o <output_dir> -t <train_size> -r <oversampling_value>
```

Here is a breakdown of the parameters:

1. `-d`: The filepath to the Dataset for input to the Machine Learning pipeline. Please see example datasets (**note**: in development)
2. `-o`: The filepath to the output directory (if it exists). If one does not then it will be created (provided the filepath is reasonably specified)
3. `-t`: A floating point size of the train test split. (For example: 0.8 does an 80:20 train test split). The default value is 0.7 i.e 70:30 split.
4. `-r`: An oversampling value. Due to the severe class imbalance it is important to specify an oversampler to help readjust the class imbalance. (Examples, 0.5 result in a 1:2 minority to majority oversampler). Also note that there can only be values between 0 and 1 so specify a floating point value that accurately compensates the pipeline and balances real world data.

## Output
Once the pipeline is run in the output folder you should see the following files:
1. `SHAP_summary_XYZ` x5: SHAP summary files of the top 20 features of each model (we have 5 ML models)
2. `test_roc`: an ROC curve on the test set showing diagnostic performance
3. `models/` : A folder which stores all models as Pickle Files
4. `processor` : A preprocessor for the ML pipeline
5. `train_test_log` : A logger which stores timestamped text on each item in the pipelines operation


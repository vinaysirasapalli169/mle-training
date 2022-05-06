# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Prerequisites:


$ conda env create --file deploy/conda/env.yml

$ conda activate mle-dev

## setup :

pip install .

## Run code:
## To download and process data:
$ python src/housing_price/ingest_data.py -r data/raw/ -p data/processed/

## To train the models:
$ python src/housing_price/train.py -d data/processed/housing_train.csv -m artifacts/

## To score trained models:
$ python src/housing_price/score.py -d data/processed/housing_test.csv -m artifacts/

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the script
python < scriptname.py >

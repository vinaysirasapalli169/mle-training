import os
import pickle
from argparse import ArgumentParser, Namespace
from glob import glob
import logging
from logging import Logger

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from house_price.logger import configure_logger


def parse_args() -> Namespace:
    
    parser = ArgumentParser()

    parser.add_argument(
        "-m",
        "--models",
        type=str,
        default="articrafts/",
        help="Directory where the models are stored.",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="data/processed/housing_test.csv",
        help="Path to test dataset csv file.",
    )

    parser.add_argument("--rmse", action="store_true", help="Show RMSE.")
    parser.add_argument("--mae", action="store_true", help="Show MAE.")

    parser.add_argument("--log-level", type=str, default="DEBUG")
    parser.add_argument("--no-console-log", action="store_true")
    parser.add_argument("--log-path", type=str, default="")

    return parser.parse_args()

def load_data(path) :
   
    df = pd.read_csv(path)
    y = df["median_house_value"].copy(deep=True)
    X = df.drop(["median_house_value"], axis=1)
    return (X, y)

def load_models(path) :
    
    paths = glob(f"{path}/*.pkl")
    paths = sorted(paths)
    models = []

    for path in paths:
        if os.path.isfile(path):
            model = pickle.load(open(path, "rb"))
            models.append(model)

    return models

def score_model(model, X,y, args) :
 
    scores = {}
    scores["R2 score"] = model.score(X, y)
    y_hat = model.predict(X)

    if args.rmse:
        rmse = np.sqrt(mean_squared_error(y, y_hat))
        scores["RMSE"] = rmse

    if args.mae:
        mae = mean_absolute_error(y, y_hat)
        scores["MAE"] = mae

    return scores


args = parse_args()
logger = configure_logger(
    log_file=args.log_path,
    log_level=args.log_level,
    console=not args.no_console_log,
)

X, y = load_data(args.dataset)

models = load_models(args.models)

for model in models:
    model_name = type(model).__name__
    scores = score_model(model, X, y, args)
    logging.debug(f"Model: {model_name}")
    for k, v in scores.items():
        logging.debug(f"{k}: {v}")

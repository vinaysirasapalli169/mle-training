import os
import pickle
from argparse import ArgumentParser, Namespace


import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import logging
from logging import Logger
from house_price.logger import configure_logger

def parse_args() :
   
    parser = ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="data/processed/housing_train.csv",
        help="Path to training dataset csv file.",
    )

    parser.add_argument(
        "-m",
        "--models",
        type=str,
        default="artifacts/",
        help="Directory to store model pickles.",
    )

    parser.add_argument("--log-level", type=str, default="DEBUG")
    parser.add_argument("--no-console-log", action="store_true")
    parser.add_argument("--log-path", type=str, default="")

    return parser.parse_args()

def load_data(path) :
    
    df = pd.read_csv(path)
    y = df["median_house_value"].copy(deep=True)
    X = df.drop(["median_house_value"], axis=1)
    return (X, y)

def save_model(model, dir) :
    
    os.makedirs(dir, exist_ok=True)
    model_name = type(model).__name__

    path = os.path.join(dir, f"{model_name}.pkl")
    with open(path, "wb") as file:
        pickle.dump(model, file)
    return (model_name, path)

args = parse_args()
logger = configure_logger(
    log_level=args.log_level,
    console=not args.no_console_log,
    log_file=args.log_path,
)

logging.info("Started training.")

X, y = load_data(path=args.dataset)

lr = LinearRegression()
lr.fit(X, y)
model_name, path = save_model(lr, args.models)

logging.debug(f"{model_name} model saved in {path}.")

dtree = DecisionTreeRegressor(random_state=42)
dtree.fit(X, y)
model_name, path = save_model(dtree, args.models)
logging.debug(f"{model_name} model saved in {path}.")

random_forest = RandomForestRegressor()
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {
        "bootstrap": [False],
        "n_estimators": [3, 10],
        "max_features": [2, 3, 4],
    },
]

grid_search = GridSearchCV(
    random_forest,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=5,
    return_train_score=True,
)
grid_search.fit(X, y)
model_name, path = save_model(grid_search.best_estimator_, args.models)
logging.debug(f"{model_name} model saved in {path}.")

logging.info("training done")


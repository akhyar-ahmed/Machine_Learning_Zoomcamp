
import argparse
import logging
import os
import random
import torch
import numpy as np
import sys
import pandas as pd

from datetime import datetime


# Read input argument
def read_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default = "asset", type = str)
    parser.add_argument("--log_folder", type = str, default = "logs")
    parser.add_argument("--save_models", action = "store_true")
    parser.add_argument("--model_name", type = str, default = "all", choices = ["all", "rfc", "lr", "xgb", "mlp"]) # default is all model
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--experiment_name", type = str, default = "")
    parser.add_argument("--experiment", type = str, default = "single_experiment", choices = ["single_experiment", "full_experiment"])
    parser.add_argument("--search_hyperparameter", type = str, default = "yes", choices = ["yes", "no"])

    return parser.parse_args()


# Create the output folder
def make_log_folder(args, name) -> str:
    if not os.path.exists(args.log_folder):
        os.mkdir(args.log_folder)

    my_date = datetime.now()

    folder_name = my_date.strftime("%Y-%m-%dT%H-%M-%S") + "_" + name

    if len(args.experiment_name) > 0:
        folder_name += "_" + args.experiment_name

    log_folder = os.path.join(args.log_folder, folder_name)
    os.mkdir(log_folder)
    return log_folder


# Write a single store to the results logfile
results_df=None
def write_results(log_dir, key, value, model_name) -> None:
    logging.info(f"write result row key={key}, value={value}")
    results_file=os.path.join(log_dir, "results.csv")

    global results_df
    if results_df is None:
        results_df=pd.DataFrame(columns = ["model", "metrics", "value"])
    
    data = [model_name, key, value]
    results_df.loc[len(results_df)] = data
    results_df.to_csv(results_file)
    return


# Log to file and console
def create_logger(log_dir) -> None:
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] {%(filename)s:%(lineno)d} %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(os.path.join(log_dir, "log.txt"))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.INFO)
    return


# Set all random seeds
def set_seed(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return


# Initialize the experiment
def init_experiments(args, experiment_name) -> str:
    log_dir = make_log_folder(args, experiment_name)
    logging.info(log_dir)
    create_logger(log_dir)
    set_seed(args.seed)

    command=" ".join(sys.argv)
    logging.info("""ML Zoomcamp: MID-TERM PROJECT (TWITTER SENTIMENT ANALYSIS)""")
    logging.info("start command: " + command)
    logging.info(f"experiment name {experiment_name}")
    logging.info(args)
    return log_dir



# Select model name
def get_model_name(name):
    if name == "rfc":
        return "RFClassifier"
    elif name == "mlp":
        return "MLPClassifier"
    elif name == "xgb":
        return "XGBClassifier"
    elif name == "lr":
        return "LogisticRegression"
    
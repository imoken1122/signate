from logs.base_log import create_logger,get_logger
from utils import load_data
import json, argparse
import pandas as pd
import numpy as np
from models.lightgbm import Lightgbm
from models.xgboost import XGBoost
from models.trainer import Trainer
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import r2_score
import pickle
import requests
import sys

CONFIG_FILE ="config_kbest_freq.json" 
K =10
VERSION = "01"
imp = True
#TASK = "classification"
TASK = "regression"
RS = 1103
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/"+ CONFIG_FILE)
    option = parser.parse_args()
    config = json.load(open(option.config))
    model_name = config["model"]
    features = config["features"]
    target_name = config["target_name"]
    params = config["model_params"]
    #lineNotify("leanrning start")
    X,y,test= load_data(features,target_name)

    #logger
    get_logger(VERSION).info(test.shape)
    get_logger(VERSION).info(option.config)
    get_logger(VERSION).info(params)
    get_logger(VERSION).info("====== CV score ======")
     
    trainer = Trainer(model_name,params,features)
    val_score,val_log = trainer.cross_validation(X,y,test,random_state = RS,n_split=K,importance_f=imp,task = TASK)
    get_logger(VERSION).info(np.mean(val_score["lgb"]))
    get_logger(VERSION).info(val_log)

    # prediction
    preds = trainer.predict()

    #submit
    trainer.create_submit(preds,val_score)


if __name__ == "__main__":
    create_logger(VERSION)
    main()

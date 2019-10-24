from logs.base_log import create_logger,get_logger
import feather
import json, argparse
import pandas as pd
import numpy as np
from models.lightgbm import Lightgbm
from models.xgboost import XGBoost
from models.trainer import Trainer
from models.stackingmodel import StackingModel
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
import pickle
import requests
import sys
from sklearn.preprocessing import StandardScaler,MinMaxScaler

CONFIG_FILE ="config_stacking.json" 
K = 2
VERSION = "01"
#TASK = "classification"
TASK = "regression"
def get_model(params):

    params["nn1"]["hidden_layer_sizes"] = tuple(params["nn1"]["hidden_layer_sizes"])
    params["nn2"]["hidden_layer_sizes"]= tuple(params["nn2"]["hidden_layer_sizes"])

    MODEL_LIST = [MLPRegressor(**params["nn1"]),
                MLPRegressor(**params["nn2"]),

                RandomForestRegressor(**params["rf1"]),
                ExtraTreesRegressor(**params["et1"]),
                 LGBMRegressor(**params["lgb1"]),
                LGBMRegressor(**params["lgb1"]),
                KNeighborsRegressor(n_neighbors=4),
                KNeighborsRegressor(n_neighbors=8),
                KNeighborsRegressor(n_neighbors=16),
                KNeighborsRegressor(n_neighbors=32),
                KNeighborsRegressor(n_neighbors=64),
                KNeighborsRegressor(n_neighbors=128),
                KNeighborsRegressor(n_neighbors=1024)]
    return MODEL_LIST

def load_data():
    std_tr = feather.read_dataframe("data/input/stack/train_standard.feather").values
    std_te = feather.read_dataframe("data/input/stack/test_standard.feather").values
    pca_tr = feather.read_dataframe("data/input/stack/tr_pca.feather").values
    pca_te = feather.read_dataframe("data/input/stack/te_pca.feather").values
    tsne_tr = feather.read_dataframe("data/input/stack/tr_tSNE.feather").values
    tsne_te = feather.read_dataframe("data/input/stack/te_tSNE.feather").values
    target = feather.read_dataframe("data/input/stack/score.feather").values

    DATA = [(std_tr,std_te),(std_tr,std_te),
            (pca_tr,pca_te),(pca_tr,pca_te),
            (pca_tr,pca_te),(tsne_tr,tsne_te),
            (std_tr,std_te),(std_tr,std_te),
            (std_tr,std_te), (std_tr,std_te),
            (std_tr,std_te),(std_tr,std_te),
            (std_tr,std_te)]
    return DATA,target

def save_feature(result):
    for name in result.keys():
        train,test = result[name][0],result[name][1]
        train,test = pd.DataFrame(train,columns=[name]),pd.DataFrame(test,columns=[name])
        train.to_feather(f"features/stack_feature/train_{name}.feather")
        test.to_feather(f"features/stack_feature/test_{name}.feather")
def main():
    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/"+ CONFIG_FILE)
    option = parser.parse_args()
    config = json.load(open(option.config))
    params = config["model_params"]
    #lineNotify("leanrning start")

    MODEL_LIST = get_model(params)
    DATA,y = load_data()
    #logger
    get_logger(VERSION).info(option.config)
    get_logger(VERSION).info(params)
    get_logger(VERSION).info("====== Stacking ======")

    model = StackingModel()
    [model.stack((m,d[0],d[1])) for m,d in zip(MODEL_LIST,DATA)]

    result,log = model.fit(y.ravel(),seed = 1103,n_splits= K)
    get_logger(VERSION).info(log)
    #save_feature(result)
    


if __name__ == "__main__":
    create_logger(VERSION)
    main()

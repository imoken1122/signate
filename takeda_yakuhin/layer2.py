

import lightgbm as lgb
import xgboost as xgb
import pandas as pd
from logs.base_log import create_logger,get_logger
import numpy as np
import matplotlib.pyplot as plt
from models.trainer import Trainer
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import r2_score
from utils.loss_function import RMSLE,CB_softmax_entorpy, KL_loss, outlier_loss,r2_coef,CB_facal_loss
import sys
import feather
"""lgb_param = {
           "boosting_type": "gbdt",
           "objective": "regression",
           "metric": "None",
           "learning_rate": 0.008,
           "min_data_in_leaf":1000,
           "bagging_fraction" : 0.8,
           "feature_fraction" : 0.8,
           "bagging_seed" : 1103,
           "verbosity": -1,
           "lambda_l1":1,
           "lambda_l2":1,

           "seed":1103
       }"""
lgb_param = {
           "boosting_type": "gbdt",
           "objective": "regression",
           "metric": "None",
           "learning_rate": 0.01,
           "max_depth":4,
          "num_leaves":10,
           "min_data_in_leaf":40,
           "bagging_fraction" : 0.6,
           "feature_fraction" : 0.3,
           "bagging_seed" : 1103,
           "verbosity": -1,
           "seed":1103
       }
xgb_param ={
        "objective": "reg:linear", 
        "eval_metric": "rmse",
        "eta":0.01,
        #"max_depth":5,
        "min_child_weight":90,
        "subsample": 0.7,
        "random_state": 1103, 
        "silent": 1,
        "seed":1103
       }

CONFIG_FILE ="config_kbest_meta.json" 

VERSION = "01"
#TASK = "classification"
TASK = "regression"
RS = 1103
n_split = 10
average = True
model_name = "xgb"
def importance(model):
    gain = model.feature_importance('gain')
    featureimp = pd.DataFrame({'feature':model.feature_name(), 
    'split':model.feature_importance('split'), 
    'gain': gain}).sort_values('gain', ascending=False)
    lgb.plot_importance(model, figsize=(50,100 ))
    plt.tight_layout()
    plt.savefig(f"./features/importance/graph.png")
    plt.show()
    print(featureimp["feature"].values)


def validate(X,y,test,seed,feature):
    cv_preds,cv_scores,log=[],[],[]
    lgb_param = {
           "boosting_type": "gbdt",
           "objective": "regression",
           "metric": "None",
           "learning_rate": 0.01,
           "max_depth":4,
          "num_leaves":10,
           "min_data_in_leaf":40,
           "bagging_fraction" : 0.6,
           "feature_fraction" : 0.3,
           "bagging_seed" : 1103,
           "verbosity": -1,
           "seed":seed
       }
    xgb_param ={
        "objective": "reg:linear", 
        "eval_metric": "rmse",
        "eta":0.1,
        "max_depth":3,
        "min_child_weight":62,
        #"colsample_bytree":0.8,
        "colsample_bylevel":0.3,
        "subsample": 0.9,
        "gamma":0.3,
        "lambda":1,
        "alpha":0,
        "random_state": 1103, 
        "silent": 1,
        "seed":seed
       }
    kf = KFold(n_splits = n_split,random_state=seed,shuffle = False)
    for i,(tr_idx,val_idx) in enumerate(kf.split(X)):
        train,valid = (X[tr_idx,:],y[tr_idx]), (X[val_idx,:],y[val_idx])
        if model_name == "lgb":
            lgb_tr = lgb.Dataset(train[0],train[1],feature_name=feature)
            lgb_val = lgb.Dataset(valid[0],valid[1],feature_name=feature)
            evals_result = {}
            #if i % 2 == 0:
            #    feval = r2_coef
            #else: feval = CB_softmax_entorpy

            model = lgb.train(
                lgb_param, 
                lgb_tr,
                valid_sets = [lgb_tr,lgb_val],
                valid_names =["train","val"],
                num_boost_round = 20000,
                early_stopping_rounds =100,
                evals_result=evals_result, 
                feval = r2_coef,
                verbose_eval=150
                #callbacks=[lgbm_logger(self.VERSION)]
            )
            print(r2_score(valid[1],model.predict(valid[0])))
            cv_pred = model.predict(test,num_iteration=model.best_iteration)
        else:
            xgb_tr = xgb.DMatrix(train[0],label = train[1])
            xgb_val = xgb.DMatrix(valid[0],label = valid[1])
            xgb_test = xgb.DMatrix(test)
            evals_result = {}
            model = xgb.train(
                xgb_param,
                xgb_tr,
                evals = [(xgb_tr,"train"),(xgb_val, "val")],
                maximize=False,
                feval = r2_coef,
                evals_result= evals_result,
                num_boost_round=20000,
                early_stopping_rounds=300,
                verbose_eval = 150,
            )
            cv_pred = model.predict(xgb_test,ntree_limit=model.best_ntree_limit)
        cv_preds.append(cv_pred.tolist())
        n_metric = list(evals_result["val"].keys())[0]
        result = np.array(evals_result["val"]["r2"])
        log.append(abs(np.min(result)))
    #importance(model)
    return np.mean(cv_preds,axis = 0),log


import os
def df_concat(flist):
    add_columns = flist[0].columns.values.tolist()
    add_df = flist[0].values
    for i in flist[1:]:
        add_columns += i.columns.values.tolist()
        add_df = np.c_[add_df,i.values]

    return pd.DataFrame(add_df,columns=add_columns)
def get_stack(path):
    files = os.listdir(path)
    if '.DS_Store' in files:
        del files[files.index('.DS_Store')]
    stack = pd.DataFrame()
    for i in files:
        tmp = feather.read_dataframe(path + i)
        stack[tmp.columns] = tmp
    return stack


def load_data():
    path1 = "features/stack_feature4/train/"
    path2 = "features/stack_feature4/test/"
    train = feather.read_dataframe("data/input/layer3/train_stack_raw.feather")
    test = feather.read_dataframe("data/input/layer3/test_stack_raw.feather").values
    score = feather.read_dataframe("data/input/stack/score.feather").values
    #train = get_stack(path1)
    #test = get_stack(path2)
    return train,test,score.ravel()

    
def main():
    get_logger(VERSION).info(xgb_param) 
    get_logger(VERSION).info("====== Stacking ======")
    train,test,score = load_data()
    f = train.columns.values.tolist()
    train = train.values
    get_logger(VERSION).info(f)
    random_avg_pred,random_score = [],[]
    
    get_logger(VERSION).info(train.shape)
    seed_list1 = [615,1122]
    if average:
        seed_list2 = [13,1122,315,615,1221,1103]
    else: seed_list2 = [13]

    for seed in seed_list2:
        preds,log = validate(train,score,test,seed,f)
        random_avg_pred.append(preds)
        random_score.append(np.mean(log))
        get_logger(VERSION).info(log)
        get_logger(VERSION).info(np.mean(log))
    print(np.mean(random_score))
    sub = pd.read_csv("data/input/sample_submit.csv",header = None)
    if average:
        sub.iloc[:,1] = np.mean(random_avg_pred,axis = 0)
    else: 
        sub.iloc[:,1] = random_avg_pred[0]
    if model_name == "lgb":
        sub.to_csv("data/output/stack_output_layer3/LGBbags_stack_raw.csv",header = None,index = None)

    else:
        sub.to_csv("data/output/XGBbags_stack_submit.csv",header = None,index = None)


if __name__ == "__main__":
    create_logger(VERSION)
    main()

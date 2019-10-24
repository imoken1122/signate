
import lightgbm as lgb
import pandas as pd
import numpy as np
import json, argparse
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,log_loss
import matplotlib.pyplot as plt
from utils.loss_function import CB_softmax_entorpy, KL_loss, RMSLE, outlier_loss,r2_coef,CB_facal_loss
import feather
CONFIG_FILE ="config_stacking.json" 
K = 10
RS = 1103
model_name = "LightGBM" + "meta"
average = True
VERSION = "01"
param = {"boosting_type": "gbdt",
            "objective": "regression",
            "metric": "None",
            "learning_rate": 0.01,
            "max_depth":20,
            "min_data_in_leaf":55,
            "bagging_fraction" : 0.6,
            "feature_fraction" : 0.3,
            "bagging_seed" : 1103,
            "lambda_l2":35,
            "max_bin":255,
            "verbosity": -1}
def LGBM(train,valid,seed):
    param = {
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
    #metrics = KL_loss if loss_f == "KL-loss" else CB_softmax_entorpy 
    metrics = r2_coef
    lgb_tr = lgb.Dataset(train[0],train[1])
    lgb_val = lgb.Dataset(valid[0],valid[1])
    evals_result = {}
    model = lgb.train(
        param, 
        lgb_tr,
        valid_sets = [lgb_tr,lgb_val],
        valid_names =["train","val"],
        num_boost_round =20000,
        early_stopping_rounds = 200,
        evals_result=evals_result,
        feval = metrics,
        verbose_eval=200,
        #callbacks=[lgbm_logger(self.VERSIO
        )
    print(r2_score(valid[1],model.predict(valid[0])))
    return model
def validation(X,y,test,seed):
    kf = KFold(n_splits=K,random_state = RS).split(X)
    next_train = np.zeros((X.shape[0],))
    fold_test = []
    log = []
    for tr_idx,val_idx in kf:     
        train,valid= (X[tr_idx],y[tr_idx]),(X[val_idx],y[val_idx])
        model = LGBM(train,valid,seed)
        next_train[val_idx] = model.predict(valid[0])
        fold_test.append((model.predict(test)).tolist())
    next_test = np.mean(fold_test,axis = 0)
    return next_train, next_test
    save_feature(next_train,next_test,model_name)


def save_feature(train,test,name):
    train,test = pd.DataFrame(train,columns=[name]),pd.DataFrame(test,columns=[name])
   # train.to_feather(f"features/stack_featur5/train/train_{name}.feather")
   # test.to_feather(f"features/stack_feature5/test/test_{name}.feather")

    train.to_feather(f"features/stack_feature_layer2/train/train_{name}.feather")
    test.to_feather(f"features/stack_feature_layer2/test/test_{name}.feather")



def load_data():
    #pca_tr = feather.read_dataframe("data/input/stack/tr_q.feather").values
    #pca_te = feather.read_dataframe("data/input/stack/te_pca.feather").values
    #allst_tr = feather.read_dataframe("data/input/stack/tr_allst.feather").values
    #allst_te = feather.read_dataframe("data/input/stack/te_allst.feather").values
    #umap_tr = feather.read_dataframe("data/input/stack/train_umap.feather").values
    #umap_te = feather.read_dataframe("data/input/stack/test_umap.feather").values
    #tsne_tr = feather.read_dataframe("data/input/stack/train_daetsne.feather").values
    #tsne_te = feather.read_dataframe("data/input/stack/test_daetsne.feather").values
    meta_tr = feather.read_dataframe("data/input/layer3/train_stack_raw.feather").values
    meta_te = feather.read_dataframe("data/input/layer3/test_stack_raw.feather").values
    #hist_tr = feather.read_dataframe("data/input/stack/train_histgram.feather").values
    #hist_te = feather.read_dataframe("data/input/stack/test_histgram.feather").values
    target = feather.read_dataframe("data/input/stack/score.feather").values
    #DATA = [(pca_tr,pca_te),(allst_tr,allst_te)]
    DATA = [(meta_tr,meta_te)]
    return DATA,target.ravel()

d,y = load_data()
tr,te = d[0][0],d[0][1]
#loss_f = ["SELoss","KL-loss"]
if average:
    seed_list = [13,1122,315,615,1221,1103]
else: 
    seed_list = [13]
tr_preds,te_preds = [],[]
for s in seed_list:
    tr_pred,te_pred = validation(tr,y,te,s)
    tr_preds.append(tr_pred)
    te_preds.append(te_pred)

save_feature(np.mean(tr_preds, axis = 0), np.mean(te_preds,axis = 0), model_name)
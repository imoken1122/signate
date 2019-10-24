from logs.base_log import create_logger,get_logger
from utils import load_data
import json, argparse
import pandas as pd
import numpy as np
from models.lightgbm import Lightgbm
from models.xgboost import XGBoost
from models.trainer import Trainer
from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold
from sklearn.metrics import r2_score
import pickle
import requests
import sys
import feather
from imblearn.under_sampling import NearMiss,RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler,SMOTE
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn import base
from tqdm import tqdm
from sklearn.model_selection import KFold
class KFoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, colnames,targetName,n_fold=3,k =5,verbosity=True,discardOriginal_col=False):

        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col
        self.k = k
    def fit(self, X, y=None):
        return self
    def transform(self,X):

        assert(type(self.targetName) == str)
        assert(type(self.colnames) == str)
        assert(self.colnames in X.columns)
        assert(self.targetName in X.columns)

        mean_of_target = X[self.targetName].mean()
        target = X[self.targetName]
        kf = StratifiedKFold(n_splits = self.n_fold, shuffle = False, random_state=1103)
        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
        X[col_mean_name] = np.nan

        for tr_ind, val_ind in kf.split(X,target):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
#             print(tr_ind,val_ind)
        #X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)[self.targetName].mean())
            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(self.fit_smoothing(X_tr, self.colnames))
        X[col_mean_name].fillna(mean_of_target, inplace = True)

        if self.verbosity:
            encoded_feature = X[col_mean_name].values
            print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,
                                                                                      self.targetName,
                                                                                      np.corrcoef(X[self.targetName].values, encoded_feature)[0][1]))
        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis=1)
            
        return X[col_mean_name]
    def fit_smoothing(self,df,col):
        """
        一つの変数に対するTarget_Encoding
        col : TargetEncodingしたい変数名
        """

        k = self.k
        n_i = df.groupby(col).count()[self.targetName]

        lambda_n_i = self.sigmoid(n_i, k)
        uni_map = df.groupby(col).mean()[self.targetName]

        return lambda_n_i * df.loc[:, self.targetName].mean() + (1 - lambda_n_i) * uni_map
    
    def sigmoid(self, x, k):
        return 1 / (1 + np.exp(- x / k))

class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self,train,colNames,encodedName):
        
        self.train = train
        self.colNames = colNames
        self.encodedName = encodedName
        
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        mean =  self.train[[self.colNames,
                self.encodedName]].groupby(
                                self.colNames).mean().reset_index() 
        dd = {}
        for index, row in mean.iterrows():
            dd[row[self.colNames]] = row[self.encodedName]
        X[self.encodedName] = X[self.colNames]
        X = X.replace({self.encodedName: dd})
        return X[self.encodedName]
def train_target_encoding(data,int_c):
    new = pd.DataFrame()
    
    for i in tqdm(int_c):
        targetc = KFoldTargetEncoderTrain(i,"Score",n_fold=10,k= 100,verbosity=False)
        new[i+'_Kfold_Target_Enc'] = targetc.fit_transform(data)
    return new
def test_target_encoding(tr,te,int_c):
    new = pd.DataFrame()
    for i in tqdm(int_c):
        test_targetc = KFoldTargetEncoderTest(tr,i,i + '_Kfold_Target_Enc')
        new[i + '_Kfold_Target_Enc'] = test_targetc.fit_transform(te)
    return new

lgbm_params = {
    'learning_rate': 0.1,
    "min_data_in_leaf":55,
    "learning_rate": 0.01,
    #"min_child_weight":23,
    "max_depth":9,
    "num_leaves":50,
    "bagging_fraction" : 0.8,
    "feature_fraction" : 0.01,
    "bagging_seed" : 1103,
    "verbosity": -1,
    "seed":1103,
    #"scale_pos_weight":3,
    "is_unbalance":True,
    'boosting_type' : 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
}

def lgbm_train(X_train_df, X_valid_df, y_train_df, y_valid_df, lgbm_params):
    lgb_train = lgb.Dataset(X_train_df, y_train_df)
    lgb_eval = lgb.Dataset(X_valid_df, y_valid_df, reference=lgb_train)
    result = {}
    # 上記のパラメータでモデルを学習する
    model = lgb.train(lgbm_params, lgb_train,
                      # モデルの評価用データを渡す
                      valid_sets=[lgb_train,lgb_eval],
                      # 最大で 1000 ラウンドまで学習する
                      num_boost_round=10000,
                      verbose_eval=200,
                      evals_result = result,
                      # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
                      early_stopping_rounds=200)
    
    return model,result
def imbalanced_data_split(X, y, test_size=0.2):
    sss = StratifiedShuffleSplit(n_splits=10,random_state=1103)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        return X_train, X_test, y_train, y_test
import json
with open('./configs/Kbest_FE.json') as f:
    df = json.load(f)
cate_c = df["kbest_feature"]["cate"]

# for validation

train = feather.read_dataframe("data/input/tr_best_class.feather")
te = feather.read_dataframe("data/input/te_best_class.feather")
y = train["Score"].values
del train["Score"],te["index"],train["index"]

#train = train.values
#X_train2, X_valid, y_train2, y_valid = imbalanced_data_split(train, y, test_size=0.2)
#sampler = RandomUnderSampler(random_state = 1103)
#model_normal = lgbm_train(X_train2, X_valid, y_train2, y_valid, lgbm_params)
count = y.sum()
bagging_pred,bag_test_preds,sc = [],[],[]
for i in range(10):
    sampler = RandomUnderSampler(ratio = {0:count*20, 1:count}, random_state = i)
    X_resampled, y_resampled = sampler.fit_resample(train.values, y)

    tr = pd.DataFrame(X_resampled,columns = train.columns)

    #tr_tgenc_f = train_target_encoding(tr.iloc[:,:],cate_c)
    #te_tgenc_f = test_target_encoding(tr,te,cate_c)
    #FE10_c = tr_tgenc_f.append(te_tgenc_f)
    #FE10_c = FE10_c.reset_index()
    #FE10_c.to_feather(f"./features/target_encode.feather")
    #del tr["Score"],train["Score"]
    smpx,smpy= tr.values,y_resampled
    # for validation
    #X_train2, X_valid, y_train2, y_valid = imbalanced_data_split(X_resampled, y_resampled, test_size=0.2)
    print(len(y_resampled[y_resampled == 0]),len(y_resampled[y_resampled == 1]))
    preds,test_preds,results = [],[],[]
    n = 5
    kf = StratifiedKFold(n_splits =n,random_state=1103)
    for i,(tr_idx,val_idx) in enumerate(kf.split(smpx,smpy)):
        X_train2, X_valid, y_train2, y_valid= smpx[tr_idx],smpx[val_idx],smpy[tr_idx],smpy[val_idx]
        model_under_sample,result = lgbm_train(X_train2, X_valid, y_train2, y_valid, lgbm_params)
        pred = model_under_sample.predict(train,num_iteration=model_under_sample.best_iteration)
        test_preds.append(model_under_sample.predict(te,num_iteration=model_under_sample.best_iteration).tolist())
        preds.append(pred.tolist())
        results.append(np.max(result['valid_1']["auc"]))
    bagging_pred.append(np.sum(preds,axis = 0)/n)
    bag_test_preds.append( np.sum(test_preds,axis = 0)/n)
    sc.append(np.mean(results))
print(np.mean(sc))
new = pd.DataFrame()
#new["test_label"] = np.sum(test_preds,axis = 0)/10
#new["target"] = np.sum(preds,axis = 0)/10
new["test_label"] = np.sum(bag_test_preds,axis = 0)/10
#new["target"] = np.sum(bagging_pred,axis = 0)/5
new.to_csv(f"pred.csv",index = None,header=None)

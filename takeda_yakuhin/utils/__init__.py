import pandas as pd
import numpy as np
import feather
from imblearn.over_sampling import SMOTE
def load_data(features,target_name):
    #train = pd.read_csv("data/input/train.csv")
    #test = pd.read_csv("data/input/test.csv")
    train = feather.read_dataframe("data/input/tr_best_prob.feather")
    test = feather.read_dataframe("data/input/te_best_prob.feather")

    target = train[target_name].values
    #target = (target - min(target))/(max(target)-min(target))
    #test_id = test["ID"].values
    #target = (target -min(target))/(max(target) - min(target))
    train,test = train[features].values,test[features].values
    
    return train, target ,test

def class_label(tgt):
    return np.where(((tgt < 0.2) | (tgt > 3.7)),1,0)

def over_sampler(tr,target):
    y = class_label(target)
    sm = SMOTE(random_state = 1103, ratio = 1.0)
    new_tr,new_y = sm.fit_sample(tr,y)

    return new_tr
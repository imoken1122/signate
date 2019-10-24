from catboost import CatBoost
from catboost import Pool
import pandas as pd
import numpy as np
from models.base_model import Model
from sklearn.metrics import r2_score,log_loss
import matplotlib.pyplot as plt

class CatBoosting(Model):
    def train_and_predict(self,train,valid,test,param,colum):
        cat_tr = Pool(train[0],label = train[1])
        cat_val = Pool(valid[0],label = valid[1])
        cat_test = Pool(test)
        model = CatBoost(param)
        model.fit(cat_tr,
                eval_set = [cat_tr,cat_val],
                early_stopping_rounds = 100,
                verbose_eval = 150,
                
                    )
        pred = model.predict(cat_test)
        print(get_evals_result())
        return model,pred,get_evals_result()

import xgboost as xgb
import pandas as pd
import numpy as np
from models.base_model import Model
from sklearn.metrics import r2_score
from utils.loss_function import CB_softmax_entorpy, outlier_loss,r2_coef
class XGBoost(Model):
    def train_and_predict(self,train,valid,test,param,f,X):
        xgb_tr = xgb.DMatrix(train[0],label = train[1])
        xgb_val = xgb.DMatrix(valid[0],label = valid[1])
        xgb_test = xgb.DMatrix(test)
        xgb_X= xgb.DMatrix(X)
        evals_result = {}
        model = xgb.train(
            param,
            xgb_tr,
            evals = [(xgb_tr,"train"),(xgb_val, "val")],
            maximize=False,
            feval = lambda pred,train : [#CB_facal_loss(pred,train)],
                                        CB_softmax_entorpy(pred,train)],
                                        #outlier_loss(pred,train)],
                                        #r2_coef(pred,train)],
            evals_result= evals_result,
            num_boost_round=10000,
            early_stopping_rounds=200,
            verbose_eval = 150,
        )
        prediction = model.predict(xgb_test,ntree_limit=model.best_ntree_limit)
        tr_pred = model.predict(xgb_X,ntree_limit=model.best_ntree_limit)
        n_metric = list(evals_result["val"].keys())[0]
        tr_result = abs(np.array(evals_result["train"][n_metric]))
        te_result = abs(np.array(evals_result["val"][n_metric]))
        return model,prediction,tr_pred,(tr_result,te_result,n_metric)




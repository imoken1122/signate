import pandas as pd
import numpy as np
from models.lightgbm import Lightgbm
from models.xgboost import XGBoost
from models.catboost import CatBoosting
from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold
from sklearn.metrics import r2_score

model_instance = {"lgb":Lightgbm(),
                  "xgb":XGBoost(), 
                  "cat":CatBoosting()
                }

class Trainer():
    def __init__(self,models_name,params,features,ensmble = False):
        self.model = {n:model_instance[n] for n in models_name}
        self.models_name = models_name
        self.preds ={n:[] for n in self.models_name}
        self.scores = {n:[] for n in self.models_name}
        self.logger ={n:[] for n in self.models_name} 
        self.learned_models = {n:[] for n in self.models_name}
        self.imp_f ={n:[] for n in self.models_name}
        self.params = params
        self.features = features
        self.ensmble = ensmble
        self.tr_preds ={n:[] for n in self.models_name}
    def cross_validation(self,X,y,test,n_split,shuffle = False,random_state =1103,importance_f = False,task = "regression"):
        label=np.where((0.5<=y) & (y<=3.8),1,0)
        if task == "regression":
            for m in self.models_name:
                cv_tr_preds,cv_preds,cv_scores,cv_logs=[],[],[],[]
                kf = KFold(n_splits = n_split,random_state=random_state,shuffle = False)
                for i,(tr_idx,val_idx) in enumerate(kf.split(X)):
                    train,valid = (X[tr_idx,:],y[tr_idx]), (X[val_idx,:],y[val_idx])
                    model_param = self.params[m]
                    learned_model,cv_pred,cv_tr_pred,result = self.model[m].train_and_predict(train,valid,test,model_param,self.features,X)
                    train_r2,val_r2,metric = result[0],result[1],result[2]
                    cv_preds.append(cv_pred.tolist())
                    cv_tr_preds.append(cv_tr_pred.tolist())
                    cv_scores.append(val_r2[-1])
                    cv_logs.append(f"[{i}]{metric} => train:{train_r2[-1]}   val:{val_r2[-1]}")
                    self.learned_models[m].append(learned_model)
                self.preds[m] = np.sum(cv_preds,axis = 0)/n_split
                self.scores[m] = cv_scores
                self.logger[m] = cv_logs
                self.tr_preds[m] = np.sum(cv_tr_preds,axis = 0)/n_split 
            if importance_f == True:
                self.get_importance()

            return self.scores,self.logger
        else : #classification
            for m in self.models_name:
                cv_tr_preds,cv_preds,cv_scores,cv_logs=[],[],[],[]
                kf = StratifiedKFold(n_splits = n_split,random_state=random_state)
                for i,(tr_idx,val_idx) in enumerate(kf.split(X,y)):
                    train,valid = (X[tr_idx,:],y[tr_idx]), (X[val_idx,:],y[val_idx])
                    model_param = self.params[m]
                    learned_model,cv_pred,cv_tr_pred,result = self.model[m].train_and_predict(train,valid,test,model_param,self.features,X)
                    train_r2,val_r2 = result[0],result[1]
                    cv_preds.append(cv_pred.tolist())
                    cv_tr_preds.append(cv_tr_pred.tolist())
                    cv_scores.append(val_r2[-1])
                    cv_logs.append(f"[{i}]{m} => train:{train_r2[-1]}   val:{val_r2[-1]}")
                    self.learned_models[m].append(learned_model)
                self.preds[m] = np.sum(cv_preds,axis = 0)/n_split
                self.scores[m] = cv_scores
                self.logger[m] = cv_logs
                self.tr_preds[m] = np.sum(cv_tr_preds,axis = 0)/n_split 
            if importance_f == True:
                self.get_importance()
            return self.scores, self.logger
    def predict(self):
        new = pd.DataFrame()
        new["target"] = self.tr_preds["lgb"]
        new.to_csv(f"tr_pred.csv",index = None,header=None)
        return self.preds
    def get_importance(self,):
        for n in self.models_name:
            score = self.scores[n]
            max_score_idx = np.argmax(score)
            opt_model = self.learned_models[n][max_score_idx]
            imp_f = self.model[n].importance(opt_model,np.mean(score))
            self.logger[n].append(imp_f["feature"].values.tolist())
            self.logger[n].append(imp_f["gain"].values.tolist())

    def create_submit(self,pred,val_score,weight = [1]):
        sub = pd.read_csv("./data/input/sample_submit.csv",header=None)
        mean_score,divide,y_sub = 0,1,0
        if self.ensmble == True:
            weight = [0.8,0.2]
            divide = len(weight)

        for w,n in zip(weight, self.models_name):
            y_sub += w * np.array(pred[n])
            mean_score += np.mean(val_score[n])

        print(mean_score)
        sub.iloc[:,1] = y_sub
        sub.to_csv(f"./data/output/submit_{round(mean_score/divide,4)}.csv",index = None,header=None)
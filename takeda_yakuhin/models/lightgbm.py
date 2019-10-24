import lightgbm as lgb
import pandas as pd
import numpy as np
from models.base_model import Model
from logs.lgbm_log import lgbm_logger
from sklearn.metrics import r2_score,log_loss
import matplotlib.pyplot as plt
from utils.loss_function import CB_softmax_entorpy, KL_loss,KU_loss, outlier_loss,r2_coef,CB_facal_loss
class Lightgbm(Model):

    def train_and_predict(self,train,valid,test,param,colum,X):
        #self.label = label
        lgb_tr = lgb.Dataset(train[0],train[1],feature_name=list(colum))
        lgb_val = lgb.Dataset(valid[0],valid[1],reference=lgb_tr,feature_name=list(colum))
        evals_result = {}
        model = lgb.train(
            param, 
            lgb_tr,
            valid_sets = [lgb_tr,lgb_val],
            valid_names =["train","val"],
            num_boost_round =20000,
            early_stopping_rounds = 700,
            evals_result=evals_result,
            #feval = self.r2_coef,
            feval = self.metrics,
            verbose_eval=200,
            #callbacks=[lgbm_logger(self.VERSION)]
        )
        prediction = model.predict(test,num_iteration=model.best_iteration)
        tr_pred = model.predict(X,num_iteration=model.best_iteration)
        print(r2_score(valid[1],model.predict(valid[0])))
        n_metric = list(evals_result["val"].keys())[0]
        tr_result = abs(np.array(evals_result["train"][n_metric]))
        te_result = abs(np.array(evals_result["val"][n_metric]))
        #tr_result = np.array(evals_result["train"]["auc"])
        #te_result = np.array(evals_result["val"]["auc"])
        return model,prediction,tr_pred,(tr_result,te_result,n_metric)

    def importance(self,model,score):
        gain = model.feature_importance('gain')
        featureimp = pd.DataFrame({'feature':model.feature_name(), 
                   'split':model.feature_importance('split'), 
                   'gain': gain}).sort_values('gain', ascending=False)
        lgb.plot_importance(model, figsize=(100,200 ))
        plt.tight_layout()
        plt.savefig(f"./features/importance/graph_{score}.png")
        return featureimp

    def metrics(self, preds, data):

        return [KU_loss(preds,data)]
            #KL_loss(preds,data)]
            #CB_facal_loss(preds,data)]
                 #CB_softmax_entorpy(preds,data)],
                    #outlier_loss(preds,data)]
                 #r2_coef(preds,data)]
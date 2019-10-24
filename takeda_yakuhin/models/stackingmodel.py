import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import r2_score
from tqdm import tqdm
import multiprocessing as mp
class StackingModel():
    def __init__(self):
        self.model_stacks = []
        self.result = {}

    def stack(self,config):
        """
        config : (model(**param),train,test)
        """
        self.model_stacks.append(config)

    def validation(self,kf):

        tr_idx,val_idx = kf
        tr_x,val_x = self.X[tr_idx],self.X[val_idx]
        tr_y,val_y = self.y[tr_idx],self.y[val_idx]

        self.model.fit(tr_x,tr_y)
        self.next_train[val_idx] = self.model.predict(val_x)
        score = r2_score(val_y,self.model.predict(val_x))
        print(f"{type(self.model).__name__} : {score}")
        self.logger.append(f"{self.model_name} : {score}")
        return self.model.predict(self.test)
      
    def fit(self,y,seed,n_splits):
        self.logger = []
        for i,cfg in enumerate(self.model_stacks):
            model,X,test = cfg[0],cfg[1],cfg[2]
            model_name = type(model).__name__ + "1"
            self.X,self.test,self.y,self.model,self.model_name = X,test,y,model,model_name
            if model_name in list(self.result.keys()):
                model_name = model_name[:-1] + str(int(model_name[-1])+1)
            n,m = X.shape[0],X.shape[1]
            self.next_train = np.zeros((n,))
            #self.next_test = np.zeros((n,))
            #self.fold_test = np.zeros((n_splits,len(test))) 
            print(f"{model_name} learning start...")
            kf = KFold(n_splits=n_splits,random_state =seed).split(X)
            
            with mp.Pool(3) as pool:
                imap = pool.map(self.validation,kf)
                fold_test = list(tqdm(imap,total = 4))

            next_test = np.mean(fold_test,axis = 0)
            self.save_feature(self.next_train,next_test,model_name)
            rst = (self.next_train,next_test)
            self.result[model_name] = rst

        return self.result,self.logger

    def save_feature(self,train,test,name):
        train,test = pd.DataFrame(train,columns=[name]),pd.DataFrame(test,columns=[name])
        train.to_feather(f"features/stack_feature/train_{name}.feather")
        test.to_feather(f"features/stack_feature/test_{name}.feather")




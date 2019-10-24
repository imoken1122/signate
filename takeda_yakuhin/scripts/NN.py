

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge,ElasticNet
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
import pandas as pd
import feather
import multiprocessing as mp
name = "EXbags"
tr_preds = np.zeros((13731,))
stack = True 
submit = False
average = True
K = 10
def yao_transform(data):
    new_yao = pd.DataFrame()
    for i in tqdm(data.columns):
        new_yao[f"{i}_yao"] = data[i].map(lambda x : -np.log1p(-x) if x < 0 else np.log1p(x))
    return new_yao
def load_data():
    #train = feather.read_dataframe("data/input/stack/train_rank.feather").values
    #test = feather.read_dataframe("data/input/stack/test_rank.feather").values
    train = feather.read_dataframe("data/input/layer3/train_stack_raw.feather")
    test = feather.read_dataframe("data/input/layer3/test_stack_raw.feather")
    score = feather.read_dataframe("data/input/stack/score.feather").values
    #train = yao_transform(train).values
    #test = yao_transform(test).values
    print(train.shape)
    return train.values,test.values,score.ravel()
tr,test,y = load_data()
def validate(kf,seed):

    tr_idx,val_idx = kf
    train,valid = (tr[tr_idx,:],y[tr_idx]), (tr[val_idx,:],y[val_idx])
    #model = MLPRegressor(hidden_layer_sizes=(1024,1024,128),early_stopping=True,random_state=1103)
    #model = KNeighborsRegressor(n_neighbors=1024)
    #model = MLPRegressor(hidden_layer_sizes=(32,50,16),max_iter = 1000,early_stopping=True, random_state=seed)
    #model =ElasticNet(alpha=0.0001,l1_ratio=5,random_state = seed,max_iter = 5000)
    #model = RandomForestRegressor(n_estimators=1000,random_state = seed)
    model = ExtraTreesRegressor(bootstrap=True, n_estimators=1000, random_state = 1103)
    #model = BayesianRidge(n_iter=3000,tol=1.e-10,lambda_1=1e-10,lambda_2=1e-10)
    #model = BayesianRidge(n_iter=3000,tol=1.e-10,lambda_=1e-10,)
    #kernel = 1*RBF(length_scale=1.0)
    #model = GaussianProcessRegressor(kernel=kernel,)
    model.fit(train[0],train[1])
    cv_tr_pred = model.predict(valid[0])

    print(r2_score(valid[1],cv_tr_pred))
    if stack:
        return np.array([val_idx,model.predict(valid[0]),model.predict(test)])
    else:
        return model.predict(test)

tr_preds,te_preds= [],[]
if average:
    seed_list = [13,1122,315,615,1221,1103]
else: seed_list = [13]
for s in seed_list:
    print(s)
    kf = KFold(n_splits = K,random_state=s).split(tr)
    arg = [(a,b) for a,b in zip(kf,[s]*K)]

    with mp.Pool(4) as pool:
        pred = pool.starmap(validate,arg)

    if stack:
        next_train = np.zeros((len(tr),))
        idx = [pred[i][0] for i in range(len(pred))]
        tr_pred = [pred[i][1] for i in range(len(pred))]
        fold_test = [pred[i][2] for i in range(len(pred))]
        for i,d in zip(idx,tr_pred):
            next_train[i] = d
        next_test = np.mean(fold_test,axis = 0)
        if average:
            tr_preds.append(next_train)
            te_preds.append(next_test)
        else:
            tr_preds = next_train
            te_preds= next_test
if submit : 
    sub = pd.read_csv("data/input/sample_submit.csv",header = None)
    if average:
        sub.iloc[:,1] = np.mean(random_avg_pred,axis = 0)
    else: 
        sub.iloc[:,1] = random_avg_pred[0]
    sub.to_csv(f"data/output/stack_output/{name}_stack_raw_submit.csv",header = None,index = None)
else:
    if average:
        next_train,next_test = np.mean(tr_preds,axis= 0),np.mean(te_preds,axis= 0)
    else:
        next_train,next_test = tr_preds,te_preds
    train,test = pd.DataFrame(next_train,columns=[name]),pd.DataFrame(next_test,columns=[name])
    train.to_feather(f"features/stack_feature_layer2/train/train_{name}.feather")
    test.to_feather(f"features/stack_feature_layer2/test/test_{name}.feather")

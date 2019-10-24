import pandas as pd
import numpy as np
import re as re

from tqdm import tqdm
from features.base import Feature, get_arguments, generate_features

Feature.dir = 'features'
from sklearn import base
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
        kf = KFold(n_splits = self.n_fold, shuffle = False, random_state=1103)
        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
        X[col_mean_name] = np.nan

        for tr_ind, val_ind in kf.split(X):
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


class Poly_encoding(Feature):
    def create_features(self)
        new = pd.DataFrame()
        data = train.append(test)
        f =[x for x in data.columns if (len(np.unique(data[x])) == 2) and (if x != "Score")]
        for i in tqdm(range(len(f))):
            for j in range(i):            
                new[f"{f[i]}_{f[j]}_XOR"] = np.logical_xor(data[f[i]], data[f[j]])  

        self.train,self.test = new.iloc[:13731,:],new.iloc[13731:,:]
class Count_encoding(Feature):
    new = pd.DataFrame()
    data = train.append(test)
    for c in tqdm(data.columns):
        if c != "Score":
            v_c = data[c].value_counts()
            new[c + "_count_enc"] = data[c].map(v_c)

    self.train,self.test = new.iloc[:13731,:],new.iloc[13731:,:]
class Label_count_encording(Feature):
    new = pd.DataFrame()
    data = train.append(test)
    for c in tqdm(data.columns):
        if c != "Score":
            count_dict = data[c].value_counts()
            label_count_dict = count_dict.rank(ascending=False).astype(int)
            encoded = data[c].map(label_count_dict)
            new[c + '_labelcount_enc'] = encoded

    self.train,self.test = new.iloc[:13731,:],new.iloc[13731:,:]


class Zero_Agg(Feature):
    def create_features(self)
        data = train.drop("Score",axis = 1).append(test)
        flist = data.columns
        new = pd.DataFrame()
        new['sum_Zeros'] = (data[flist] == 0).astype(int).sum(axis=1)
        new['mean_Zeros'] = (data[flist] == 0).astype(int).mean(axis=1)
        new['median_Zeros'] = (data[flist] == 0).astype(int).median(axis=1)
        new['max_Zeros'] = (data[flist] == 0).astype(int).max(axis=1)
        new['min_Zeros'] = (data[flist] == 0).astype(int).min(axis=1)
        new['std_Zeros'] = (data[flist] == 0).astype(int).std(axis=1)
        new['var_Zeros'] = (data[flist] == 0).astype(int).var(axis=1)
        self.train,self.test = new.iloc[:13731,:],new.iloc[13731:,:]



class Neg_Agg(Feature):
    def create_features(self)
        data = train.drop("Score",axis = 1).append(test)
        flist = data.columns
        new = pd.DataFrame()
        new['sum_Inc'] = (data[flist] > 0).astype(int).sum(axis=1)
        new['mean_Inc'] = (data[flist] > 0).astype(int).mean(axis=1)
        new['median_Inc'] = (data[flist] > 0).astype(int).median(axis=1)
        new['std_Inc'] = (data[flist] > 0).astype(int).std(axis=1)
        new['var_Inc'] = (data[flist] > 0).astype(int).var(axis=1)
        new['min_Inc'] = (data[flist] > 0).astype(int).min(axis=1)
        new['max_Inc'] = (data[flist] > 0).astype(int).max(axis=1)
        self.train,self.test = new.iloc[:13731,:],new.iloc[13731:,:] 


class Target_encording(Feature):
    def create_features(self):
        train["score_label"] = self.get_label(train["Score"])
        del train["Score"]
        int_c = train.columns.values
        self.train = train_target_encoding(train,int_c)
        self.test = test_target_encoding(train,test,int_c)

    def train_target_encoding(self,data,int_c):
        new = pd.DataFrame()
        for i in tqdm(int_c):
            targetc = KFoldTargetEncoderTrain(i,"score_label",n_fold=5,k= 100,verbosity=False)
            new[i+'_Kfold_Target_Enc'] = targetc.fit_transform(data)
        return new
    def test_target_encoding(self,tr,te,int_c):
        new = pd.DataFrame()
        for i in tqdm(int_c):
            test_targetc = KFoldTargetEncoderTest(tr,i,i + '_Kfold_Target_Enc')
            new[i + '_Kfold_Target_Enc'] = test_targetc.fit_transform(te)
        return new
    def get_label(self,score):
        label = []
        for i in score:
            if i <= -1.:
                label.append(0)
            elif -1 < i <= 0:
                label.append(1)
            elif 0 < i <= 1:
                label.append(2)
            elif 1 < i <= 2:
                label.append(3)
            elif 2 < i <= 3:
                label.append(4)
            else:
                label.append(5)
        return label
def category_preprocess(train,test):

    #分散＝０
    constant = []
    for col in train.columns:
        if col != "Score" or col != "ID":
            if train[col].std() == 0:
                constant.append(col)
    train1 = train.drop(columns=constant,axis=1)
    test1 = test.drop(columns=constant,axis=1)


    # カテゴリ minが300以下なら排除
    col=[]
    for c in tqdm(train1.columns):
        if len(np.unique(train1[c])) == 2:
            if min(train1[c].value_counts()) <= 300:
                col.append(c)
        if train1[c].dtypes == float and len(np.unique(train1[c])) < 3:
            col.append(c)
            
        if train1[c].value_counts().max()/len(train1) > 0.92:
            col.append(c)
    train1.drop(col,axis = 1, inplace=True)
    test1.drop(col,axis = 1, inplace=True)
            
    #ダブり
    aa = train1.columns
    duplicate_f = train1[aa].T[train1[aa].T.duplicated()].index.values
    train1.drop(duplicate_f,axis=1,inplace=True)
    test1.drop(duplicate_f,axis=1,inplace=True)

    int_data = train1.append(test1)
    cate_c = []
    for i in int_data.columns:
        a = int_data[i].value_counts().keys().values
        aa = np.arange(min(a),max(a) + 1)
        if len(set(a)) == len(aa):
            cate_c.append(i)

    train1 = train1[cate_c]
    test1 = test1[cate_c]

    return train1,test1

if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_csv('./data/input/train.csv')
    test = pd.read_csv('./data/input/test.csv')
    del train["ID"],test["ID"]
    train,test = category_preprocess(train.drop("Score",axis = 1),test)

    generate_features(globals(), args.force)
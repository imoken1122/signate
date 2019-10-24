import pandas as pd
import numpy as np
import re as re

from tqdm import tqdm
from features.base import Feature, get_arguments, generate_features

Feature.dir = 'features'


class Importance_interact(Feature):
    def create_features(self):
    #Lihghtgbm importance feauter
        new = pd.DataFrame()
        data = train.append(test)
        flist = [x for x in data.columns if (data[x].dtypes == int) and (len(np.unique(data[x])) != 2) ]
        for c in tqdm(flist):
            new["add_" + imp_c +"_"+ c ] = data[imp_c]+ data[c]
            new["minus_" + imp_c +"_"+ c ] = data[imp_c]- data[c]
            new["dot_" + imp_c +"_"+ c ] = data[imp_c]*data[c]
        self.train,self.test = new.iloc[:13731,:],new.iloc[13731:,:]
class Unique_Agg(Feature):
    def create_features(self):
        threshold,st = 21,"int"
        data = train.append(test)
        new_arr = np.zeros((data.shape[0],data.shape[1]))
        for i in tqdm(range(len(data))):
            tmp = np.unique(data.iloc[i,:])
            tmp = tmp[tmp>0]
            if len(tmp) > 0:
                new_arr[i,:len(tmp)] = tmp
                
        col = [f"FE1_{st}{i}" for i in range(threshold)]
        new = pd.DataFrame(new_arr[:,:threshold],columns=col)
        for i in tqdm(range(len(data))):
            tmp0 = new_arr[i,:threshold]
            tmp1 = new_arr[i,threshold:]
            tmp2 = new_arr[i,:]
            
            new.loc[i,f'{st}_sum_Uniq'] = tmp2[tmp2 == 0].sum()
            new.loc[i,f'{st}_mean_Uniq'] = tmp1.mean()
            new.loc[i,f'{st}_std_Uniq'] = tmp1.std()
            new.loc[i,f'{st}_var_Uniq'] =  tmp1.var()
            new.loc[i,f'{st}_max_Uniq_af'] =  tmp1.max()
            new.loc[i,f'{st}_min_Uniq_af'] =  tmp1.min()
            new.loc[i,f'{st}_max_Uniq_be'] =  tmp0.max()
            new.loc[i,f'{st}_min_Uniq_be'] =  tmp0.min()
        
            self.train,self.test = new.iloc[:13731,:],new.iloc[13731:,:]
class Decrease_Agg(Feature):
    def create_features(self):
        threshold,st = 21,"int"
        data = train.append(test)
        new = pd.DataFrame()
        new_arr = np.zeros((data.shape[0],data.shape[1]))
        for i in tqdm(range(len(data))):
            tmp = data.iloc[i,:]
            tmp = tmp[tmp>0]
            _dict = dict(sorted(dict(tmp).items(), key=lambda x: -x[1]))
            tmp = tmp[list(_dict.keys())]
            if len(tmp) > 0:
                new_arr[i,:len(tmp)] = tmp
            
        col = [f"FE3_{st}{i}" for i in range(threshold)]
        new = pd.DataFrame(new_arr[:,:threshold],columns=col)
        for i in tqdm(range(len(data))):
            tmp0 = new_arr[i,:threshold]
            tmp1 = new_arr[i,threshold:]
            tmp2 = new_arr[i,:]
            
            new.loc[i,f'{st}_sum_Decrease'] = tmp2[tmp2 == 0].sum()
            new.loc[i,f'{st}_mean_Decrease'] = tmp1.mean()
            new.loc[i,f'{st}_std_Decrease'] = tmp1.std()
            new.loc[i,f'{st}_var_Decrease'] =  tmp1.var()
            new.loc[i,f'{st}_max_Decrease_af'] =  tmp1.max()
            new.loc[i,f'{st}_min_Decrease_af'] =  tmp1.min()
            new.loc[i,f'{st}_max_Decrease_be'] =  tmp0.max()
            new.loc[i,f'{st}_min_Decrease_be'] =  tmp0.min()
        
        self.train,self.test = new.iloc[:13731,:],new.iloc[13731:,:]
class Increase_Agg(Feature):
    def create_features(self):
        threshold,st = 21,"int"
        data = train.append(test)
        new = pd.DataFrame()
        new_arr = np.zeros((data.shape[0],data.shape[1]))
        for i in tqdm(range(len(data))):
            tmp = data.iloc[i,:]
            tmp = tmp[tmp>0]
            _dict = dict(sorted(dict(tmp).items(), key=lambda x: x[1]))
            tmp = tmp[list(_dict.keys())]
            if len(tmp) > 0:
                new_arr[i,:len(tmp)] = tmp
            
        col = [f"FE3_{st}{i}" for i in range(threshold)]
        new = pd.DataFrame(new_arr[:,:threshold],columns=col)
        for i in tqdm(range(len(data))):
            tmp0 = new_arr[i,:threshold]
            tmp1 = new_arr[i,threshold:]
            tmp2 = new_arr[i,:]
            
            new.loc[i,f'{st}_sum_Decrease'] = tmp2[tmp2 == 0].sum()
            new.loc[i,f'{st}_mean_Decrease'] = tmp1.mean()
            new.loc[i,f'{st}_std_Decrease'] = tmp1.std()
            new.loc[i,f'{st}_var_Decrease'] =  tmp1.var()
            new.loc[i,f'{st}_max_Decrease_af'] =  tmp1.max()
            new.loc[i,f'{st}_min_Decrease_af'] =  tmp1.min()
            new.loc[i,f'{st}_max_Decrease_be'] =  tmp0.max()
            new.loc[i,f'{st}_min_Decrease_be'] =  tmp0.min()
        
        self.train,self.test = new.iloc[:13731,:],new.iloc[13731:,:]
class Zero_Agg(Feature):
    def create_features(self)
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
    def create_features(self):
        data = train.append(test)
        flist = data.columns
        new = pd.DataFrame()
        new['sum_Neg'] = (data[flist] < 0).astype(int).sum(axis=1)
        new['median_Neg'] = (data[flist] < 0).astype(int).median(axis=1)
        new['mean_Neg'] = (data[flist] < 0).astype(int).mean(axis=1)
        new['std_Neg'] = (data[flist] < 0).astype(int).std(axis=1)
        new['var_Neg'] = (data[flist] < 0).astype(int).var(axis=1)
        new['max_Neg'] = (data[flist] < 0).astype(int).min(axis=1)
        new['min_Neg'] = (data[flist]< 0).astype(int).max(axis=1)
        self.train,self.test = new.iloc[:13731,:],new.iloc[13731:,:] 



class Neg_Agg(Feature):
    def create_features(self):
        data = train.append(test)
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


class AllUnique_Agg(Feature):
    def create_features(self):
        data = train.append(test)
        flist = data.columns
        new = pd.DataFrame()
        for i in tqdm(range(len(data))):
            new.loc[i,'sum_Uniq'] = np.unique(data.iloc[i,:]).sum()
            new.loc[i,'mean_Uniq'] = np.unique(data.iloc[i,:]).mean()
            new.loc[i,'std_Uniq'] = np.unique(data.iloc[i,:]).std()
            new.loc[i,'var_Uniq'] =  np.unique(data.iloc[i,:]).var()
            new.loc[i,'min_Uniq'] =  np.unique(data.iloc[i,:]).max()
            new.loc[i,'max_Uniq'] =  np.unique(data.iloc[i,:]).min()
        self.train,self.test = new.iloc[:13731,:],new.iloc[13731:,:] 

def int_preprocess(train,test):

    #分散＝０
    constant = []
    for col in train.columns:
    
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
    train1.drop(cate_c,axis = 1,inplace=True)
    test1.drop(cate_c,axis = 1,inplace=True)

    return train1,test1

if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_csv('./data/input/train.csv')
    test = pd.read_feather('./data/input/test.csv')
    del train["Score"],train["ID"],test["ID"]
    train,test = int_preprocess(train,test)

    generate_features(globals(), args.force)
import pandas as pd
import numpy as np
import re as re

from tqdm import tqdm
from base import Feature, get_arguments, generate_features

Feature.dir = 'features'


class Intract_encoding(Feature):
    def create_features(self):
    #Lihghtgbm importance feauter
        f= ['col329', 'col2431', 'col226', 'col2614', 'col3552', 'col2621', 'col3277', 'col420', 'col176', 'col248', 'col119', 'col381',
        'col3642', 'col46', 'col253', 'col1174', 'col552', 'col71',
       'col3653', 'col1306', 'col1815', 'col913', 'col1094', 'col44','col130', 'col441', 'col3431', 'col48', 'col3474', 'col2989',
       'col1050', 'col113', 'col56', 'col1252', 'col15', 'col209','col1011', 'col99', 'col1632', 'col3194', 'col124', 'col2834',
       'col398', 'col68', 'col3063', 'col104', 'col723', 'col904','col2537', 'col945', 'col1731', 'col1279', 'col1315', 'col743']
        new = pd.DataFrame()
        data = train.append(test)
        for i in tqdm(range(len(f))):
            for j in range(i):
                c,cc = f[i],f[j]
                new["add_" + c +"_"+ cc ] = data[c].values + data[cc].values
                new["minus_" + c +"_"+ cc ] = data[c].values- data[cc].values
                new["dot_" + c +"_"+ cc ] = data[c].values*data[cc].values
        self.train,self.test = new.iloc[:13731,:],new.iloc[13731:,:]
class Increase_Agg(Feature):
    def create_features(self):
        threshold = 21,"float"
        new = pd.DataFrame()
        data = train.append(test)
        new_arr = np.zeros((data.shape[0],data.shape[1]))
        for i in tqdm(range(len(data))):
            tmp = data.iloc[i,:]
            tmp = tmp[tmp>0]
            _dict = dict(sorted(dict(tmp).items(), key=lambda x: x[1]))
            tmp = tmp[list(_dict.keys())]
            if len(tmp) > 0:
                new_arr[i,:len(tmp)] = tmp
        col = [f"FE4_{st}{i}" for i in range(threshold)]
        new = pd.DataFrame(new_arr[:,:threshold],columns=col)
        for i in tqdm(range(len(data))):
            tmp0 = new_arr[i,:threshold]
            tmp1 = new_arr[i,threshold:]
            tmp2 = new_arr[i,:]
            
            new.loc[i,f'{st}_sum_Increase'] = tmp2[tmp2 == 0].sum()
            new.loc[i,f'{st}_mean_Increase'] = tmp1.mean()
            new.loc[i,f'{st}_std_Increase'] = tmp1.std()
            new.loc[i,f'{st}_var_Increase'] =  tmp1.var()
            new.loc[i,f'{st}_max_Increase_af'] =  tmp1.max()
            new.loc[i,f'{st}_min_Increase_af'] =  tmp1.min()
            new.loc[i,f'{st}_max_Increase_be'] =  tmp0.max()
            new.loc[i,f'{st}_min_Increase_be'] =  tmp0.min()
        self.train,self.test = new.iloc[:13731,:],new.iloc[13731:,:]
class Decrease_Agg(Feature):
    def create_features(self):
        threshold,st = 21,"float"
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
def float_preprocess(train,test):

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
    train1.drop(col,axis = 1, inplace=True)
    test1.drop(col,axis = 1, inplace=True)
                
    #ダブり
    aa = []
    for i in tqdm(train1.columns):
        if train1[i].dtype != "object":
            aa.append(i)
    duplicate_f = train1[aa].T[train1[aa].T.duplicated()].index.values
    train1.drop(duplicate_f,axis=1,inplace=True)
    test1.drop(duplicate_f,axis=1,inplace=True)

    #分散
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=0.1)
    sel.fit(train1)
    f = train1.columns[sel.get_support()].values
    train1 = train1[f]
    test1 = test[f]


    from scipy.stats import ks_2samp
    list_p_value =[]

    for i in tqdm(train1.columns):
        list_p_value.append(ks_2samp(test1[i] , train1[i])[1])

    Se = pd.Series(list_p_value, index = train1.columns).sort_values() 
    list_discarded = list(Se[Se < .1].index)
    train1.drop(list_discarded,axis=1,inplace=True)
    test1.drop(list_discarded,axis=1,inplace=True)
    #相関係数
    corr = set()
    corr_matrix = train1.corr()
    for i in tqdm(range(len(corr_matrix.columns))):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > 0.8:
                corr.add(corr_matrix.columns[i])
    train1.drop(corr,axis=1,inplace=True)
    test1.drop(corr,axis=1,inplace=True)

    return train1,test1

if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_csv('./data/input/train.csv')
    test = pd.read_csv('./data/input/test.csv')
    del train["Score"],train["ID"],test["ID"]
    train,test = float_preprocess(train,test)

    generate_features(globals(), args.force)
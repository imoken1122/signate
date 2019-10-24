import numpy as np
from sklearn.model_selection import KFold
from gokinjo.backend_sklearn import ScikitTransformer
from tqdm import tqdm
import feather
import json, argparse
import pandas as pd
from gokinjo import knn_kfold_extract
from sklearn.preprocessing import StandardScaler
from gokinjo import knn_extract
int_c = ['col4', 'col70', 'col88', 'col95', 'col142', 'col144', 'col153',
       'col161', 'col162', 'col195', 'col286', 'col458', 'col463',
       'col537', 'col569', 'col586', 'col588', 'col635', 'col680',
       'col746', 'col753', 'col777', 'col793', 'col821', 'col879',
       'col887', 'col922', 'col925', 'col957', 'col975', 'col986',
       'col1006', 'col1039', 'col1040', 'col1053', 'col1095', 'col1096',
       'col1138', 'col1170', 'col1184', 'col1185', 'col1195', 'col1219',
       'col1223', 'col1229', 'col1232', 'col1235', 'col1287', 'col1329',
       'col1399', 'col1444', 'col1447', 'col1459', 'col1517', 'col1541',
       'col1629', 'col1704', 'col1725', 'col1741', 'col1747', 'col1793',
       'col1857', 'col1862', 'col1889', 'col1991', 'col2000', 'col2062',
       'col2065', 'col2072', 'col2117', 'col2143', 'col2180', 'col2185',
       'col2230', 'col2241', 'col2258', 'col2261', 'col2275', 'col2281',
       'col2343', 'col2349', 'col2360', 'col2387', 'col2397', 'col2399',
       'col2410', 'col2414', 'col2428', 'col2433', 'col2486', 'col2493',
       'col2514', 'col2565', 'col2584', 'col2677', 'col2693', 'col2787',
       'col2818', 'col2845', 'col2868', 'col2981', 'col2992', 'col2998',
       'col3023', 'col3075', 'col3094', 'col3109', 'col3116', 'col3203',
       'col3215', 'col3253', 'col3312', 'col3330', 'col3331', 'col3364',
       'col3372', 'col3385', 'col3399', 'col3423', 'col3529', 'col3535',
       'col3540', 'col3551', 'col3607', 'col3632', 'col3640', 'col3684',
       'col3723', 'col3727', 'col3741', 'col3751', 'col3769', 'col3775',
       'col3788', 'col3797', 'col3805']
fl_c = ['col5', 'col10', 'col15', 'col22', 'col29', 'col38', 'col44',
       'col46', 'col48', 'col56', 'col68', 'col71', 'col87', 'col99',
       'col104', 'col106', 'col107', 'col113', 'col115', 'col118',
       'col119', 'col121', 'col124', 'col130', 'col134', 'col168',
       'col176', 'col182', 'col208', 'col209', 'col218', 'col226',
       'col232', 'col237', 'col240', 'col248', 'col253', 'col255',
       'col256', 'col269', 'col280', 'col281', 'col305', 'col306',
       'col312', 'col329', 'col340', 'col342', 'col344', 'col352',
       'col357', 'col359', 'col381', 'col413', 'col420', 'col441',
       'col453', 'col454', 'col514', 'col524', 'col525', 'col533',
       'col552', 'col554', 'col567', 'col571', 'col590', 'col598',
       'col608', 'col620', 'col624', 'col654', 'col670', 'col679',
       'col690', 'col699', 'col723', 'col730', 'col744', 'col766',
       'col767', 'col768', 'col775', 'col791', 'col803', 'col816',
       'col820', 'col900', 'col945', 'col965', 'col1008', 'col1032',
       'col1038', 'col1050', 'col1071', 'col1118', 'col1139', 'col1174',
       'col1242', 'col1252', 'col1273', 'col1279', 'col1292', 'col1306',
       'col1315', 'col1318', 'col1323', 'col1337', 'col1382', 'col1469',
       'col1532', 'col1535', 'col1540', 'col1545', 'col1555', 'col1557',
       'col1563', 'col1572', 'col1587', 'col1601', 'col1607', 'col1632',
       'col1669', 'col1687', 'col1698', 'col1731', 'col1759', 'col1768',
       'col1770', 'col1792', 'col1796', 'col1815', 'col1844', 'col1855',
       'col1873', 'col1938', 'col1960', 'col2074', 'col2077', 'col2101',
       'col2147', 'col2219', 'col2236', 'col2306', 'col2365', 'col2391',
       'col2413', 'col2431', 'col2439', 'col2518', 'col2537', 'col2560',
       'col2614', 'col2616', 'col2621', 'col2669', 'col2699', 'col2722',
       'col2744', 'col2785', 'col2789', 'col2793', 'col2834', 'col2867',
       'col2878', 'col2885', 'col2888', 'col2902', 'col2914', 'col2921',
       'col2925', 'col2963', 'col2989', 'col3051', 'col3058', 'col3139',
       'col3194', 'col3275', 'col3277', 'col3300', 'col3353', 'col3379',
       'col3431', 'col3472', 'col3477', 'col3488', 'col3521', 'col3531',
       'col3552', 'col3590', 'col3642', 'col3653', 'col3661', 'col3681',
       'col3698', 'col3699', 'col3711', 'col3725']
def df_concat(flist):
    add_columns = flist[0].columns.values.tolist()
    add_df = flist[0].values
    for i in flist[1:]:
        add_columns += i.columns.values.tolist()
        add_df = np.c_[add_df,i.values]
    return pd.DataFrame(add_df,columns=add_columns)

CONFIG_FILE ="Kbest_FE.json"
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/"+ CONFIG_FILE)
option = parser.parse_args()
config = json.load(open(option.config))
int_c = config["kbest_feature"]["int"]
fl_c = config["kbest_feature"]["float"]
pca_c = [f"tsne_{i+1}" for i in range(3)]
train = feather.read_dataframe("data/input/tr_best_class.feather")
test = feather.read_dataframe("data/input/te_best_class.feather")
#FE_int = feather.read_dataframe("features/add_float.feather")
#FE_float = feather.read_dataframe("features/add_int.feather")
#FE = df_concat([FE_int,FE_float])
label = train.Score.values
del train["index"],test["index"],train["Score"]

k = 1
data = train.append(test)
#num_data = data[pca_c]
int_data = data[int_c]
fl_data = data[fl_c]
num_data = df_concat([int_data,fl_data])
#num_data = df_concat([num_data,FE])
tr,te = num_data.iloc[:13731,:].values,num_data.iloc[13731:,:].values


number_of_classes = np.unique(label)

#stdsc = StandardScaler()
#tr = stdsc.fit_transform(tr)
#te = stdsc.transform(te)

train_knn = knn_kfold_extract(tr,label, random_state=1103,k = k,folds= 5)
test_knn = knn_extract(tr,label,te, k=k)

print(train_knn.shape,test_knn.shape)

col = [f"knn_{i}" for i in range(len(number_of_classes) * k)]
tr = pd.DataFrame(train_knn,columns = col)
te = pd.DataFrame(test_knn,columns = col)

tr = df_concat([train,tr])
te = df_concat([test,te])
tr["Score"] = label
tr.to_feather("data/input/tr_best_class.feather")
te.to_feather("data/input/te_best_class.feather")
import json
#del tr["Score"]
#ff = tr.columns.values.tolist()
#features = {"features":ff, "target_name":"Score"}

#with open("configs/config_kbest_knn.json","w") as f:
#    json.dump(features,f)
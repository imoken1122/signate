import umap
import feather
import pandas as pd
import matplotlib.pyplot as plt
#data = feather.read_dataframe("features/DAE.feather").values
train = feather.read_dataframe("data/input/tr_best.feather")
test = feather.read_dataframe("data/input/te_best.feather")
del train["Score"],train["index"],test["index"],test["ID"]
data = train.append(test).values[:,:]
trans = umap.UMAP(n_neighbors=20,n_components=3, random_state=1103).fit_transform(data)

c = [f"rawUMAP_{i}" for i in range(trans.shape[1])]
df = pd.DataFrame(trans,columns=c)
df.to_feather("features/raw_umap20.feather")
plt.scatter(trans[:, 0], trans[:, 1], s= 5)
plt.title('Embedding of the training set by UMAP', fontsize=24)
plt.show()

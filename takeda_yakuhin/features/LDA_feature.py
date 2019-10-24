import pandas as pd
import numpy as np
import re as re
import feather
from tqdm import tqdm
from base import Feature, get_arguments, generate_features
from LDA import LDA
Feature.dir = 'features'


class LDA_feature(Feature):
    def create_features(self):
        cate_col = ['col9',
                    'col609',
                    'col1317',
                    'col1430',
                    'col1975',
                    'col2135',
                    'col3213',
                    'col3289',
                    'col3290',
                    'col3519',
                    'col3591']
        
        co_feature1 = [(cate_col[i],cate_col[j]) for i in range(len(cate_col)) for j in range(i)]
        co_feature2 = [(cate_col[j],cate_col[i]) for i in range(len(cate_col)) for j in range(i)]
        co_feature = co_feature1 + co_feature2
        n_topics,n_lda_fe = 3,10
        lda = LDA(co_feature,n_topics,n_lda_fe)
        lda.fit(train)
        train_lda = lda.transform(train,cate_col)
        test_lda = lda.transform(test,cate_col)
        new_col = [f"{c}_topic{i}" for c in co_feature for i in range(n_topics)]
        for i,c in enumerate(new_col):
            self.train[c]= train_lda[:,i]
            self.test[c] = test_lda[:,i]

if __name__ == '__main__':
    args = get_arguments()

    train = feather.read_dataframe('data/input/train.feather')
    test = feather.read_dataframe('data/input/test.feather')
    del train["Score"],train["ID"],test["ID"]

    generate_features(globals(), args.force)
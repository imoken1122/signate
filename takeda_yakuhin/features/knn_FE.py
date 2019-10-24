import pandas as pd
import numpy as np
import re as re

from tqdm import tqdm
from features.base import Feature, get_arguments, generate_features

Feature.dir = 'features'


class knn_feature(Feature):
    def create_features(self):
        

if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_feather('./data/input/train.feather')
    test = pd.read_feather('./data/input/test.feather')
    del train["Score"],train[test["ID"]
    train,test = int_preprocess(train,test)

    generate_features(globals(), args.force)
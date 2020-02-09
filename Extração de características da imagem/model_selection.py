import numpy as np
import pandas as pd
from math import ceil

def train_test_split_per_class(x, y, test_size):
    test_indexes = []

    for _class in y.unique():
        indexes = y[y == _class].index.values

        test_indexes.extend(np.random.choice(indexes, ceil(test_size*indexes.size)))

    x_train = x.drop(test_indexes)
    x_test = x.iloc[test_indexes]
    y_train = y.drop(test_indexes)
    y_test = y.iloc[test_indexes]

    return x_train, x_test, y_train, y_test
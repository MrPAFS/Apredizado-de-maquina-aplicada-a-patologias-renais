import numpy as np
import pandas as pd
from math import ceil
from sklearn.metrics import accuracy_score
from statistics import mean

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

def cross_validation(model, x, y, test_size, iterations):
    accuracy_set = []

    for _ in range(iterations):
        x_train, x_test, y_train, y_test = train_test_split_per_class(x, y, test_size)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        accuracy_set.append(accuracy_score(y_test, y_pred))

    return accuracy_set, mean(accuracy_set)

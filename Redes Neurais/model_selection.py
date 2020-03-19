import numpy as np
import pandas as pd
from math import ceil
from statistics import mean
import tensorflow as tf

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
        model_copy = tf.keras.models.clone_model(model)
        model_copy.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        
        x_train, x_test, y_train, y_test = train_test_split_per_class(x, y, test_size)

        model_copy.fit(x = x_train.to_numpy(),y = y_train.to_numpy(), batch_size = 10, epochs = 20, verbose=0)
        _, acc = model_copy.evaluate(x_test.to_numpy(), y_test.to_numpy(), verbose=0)

        accuracy_set.append(acc)

    return accuracy_set, mean(accuracy_set)
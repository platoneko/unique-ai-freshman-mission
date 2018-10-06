# -*- coding: utf-8 -*-


import numpy as np


def get_dist2(a, b):
    return np.sum((a - b) ** 2)


def kNN(train_x, train_y, test_x, test_y, k=3):
    m, n  = train_x.shape
    dist = [0] * m
    for i in range(m):
        dist[i] = get_dist2(train_x[i], test_x)
    sorted_index = np.argsort(dist).tolist()
    counter = dict()
    for i in range(k):
        y = train_y[sorted_index[i]]
        if y not in counter:
            counter[y] = 0
        counter[y] += 1
    y_hat = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
    return y_hat, y_hat == test_y


def normalization(input_x):
    max_value = np.max(input_x, 0)
    min_value = np.min(input_x, 0)
    range_value = max_value - min_value
    output_x = (input_x - min_value)/(range_value + 1e-7)
    return output_x


def test(train_x, train_y, test_x, test_y, k):
    m, n = test_x.shape
    result = [0] * m
    acc = 0
    for i in range(m):
        result[i], correct = kNN(train_x, train_y, test_x[i], test_y[i], k)
        acc += correct
    acc /= m
    return result, acc


if __name__ == '__main__':
    mnist = np.load('mnist.npz')
    train_x, train_y, test_x, test_y = mnist['x_train'], mnist['y_train'], mnist['x_test'], mnist['y_test']

    train_x = np.reshape(train_x, [60000, 28*28])
    train_x = normalization(train_x)

    test_x = np.reshape(test_x, [10000, 28*28])
    test_x = normalization(test_x)

    result, acc = test(train_x, train_y, test_x, test_y, k=5)
    print("acc = {}".format(acc))


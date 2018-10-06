# -*-coding:utf-8-*-


import numpy as np
import time
import pandas as pd
from math import *


class Linear_SVC:
    def __init__(self, w, b, C, lr):
        self.w = w.copy()
        self.b = b
        self.C = C
        self.lr = lr

    def fit(self, X, y, max_iter):
        for k in range(max_iter):
            grad_w = 0
            grad_b = 0
            grad_w += self.C * self.w
            loss_mat = 1 - y * (self.w.dot(X.T) + self.b)
            loss_mat[loss_mat < 0] = 0
            loss = np.sum(loss_mat)

            activation_mat = np.zeros(X.shape[0])
            activation_mat[loss_mat > 0] = 1

            grad_w -= (y * activation_mat).dot(X)
            grad_b -= np.sum(y * activation_mat)

            if (k+1)%500 == 0:
                print("iter: {}, loss: {}".format(k+1, loss))

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def predict(self, X):
        return np.sign(X.dot(self.w) + self.b)


class Multi_Linear_SVC:
    def __init__(self, W, b, C, lr):
        self.W = W.copy()
        self.b = b.copy()
        self.C = C
        self.lr = lr

    def fit(self, X, y, max_iter):
        for k in range(max_iter):
            grad_W = 0
            grad_b = 0
            grad_W += self.C * self.W
            score_mat = X.dot(self.W) + self.b

            loss_mat = score_mat.copy() - score_mat[range(X.shape[0]), y].reshape([-1, 1]) + 1
            loss_mat[range(X.shape[0]), y] = 0
            loss_mat[loss_mat < 0] = 0
            loss = np.sum(loss_mat)

            activation_mat = np.zeros([X.shape[0], self.W.shape[1]])
            activation_mat[loss_mat > 0] = 1
            activation_mat[range(X.shape[0]), y] -= np.sum(activation_mat, 1)

            grad_W += X.T.dot(activation_mat)
            grad_b += np.sum(activation_mat, 0)

            if (k+1) % 500 == 0:
                print("iter: {}, loss: {}".format(k+1, loss))

            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b

    def predict(self, X):
        return np.argmax(X.dot(self.W) + self.b, 1)


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    dataset = load_breast_cancer()
    data = dataset['data']
    labels = dataset['target']
    labels[labels == 0] = -1

    data = (data - np.mean(data, 0)) / np.std(data, 0)

    train_X = data[:400]
    train_y = labels[:400]
    valid_X = data[400:500]
    valid_y = labels[400:500]
    test_X = data[500:]
    test_y = labels[500:]

    w = np.random.randn(30)
    b = 0
    lr = 0.0003
    C = 0.0001

    clf = Linear_SVC(w, b, C, lr)

    t3 = time.time()
    clf.fit(train_X, train_y, 100000)
    t4 = time.time()
    print("the training time is: %fs" % (t4-t3))

    predict = clf.predict(valid_X)
    acc = np.mean(predict == valid_y)
    print("acc on the validation set: ", acc)

    test_predict = clf.predict(test_X)
    test_acc = np.mean(test_predict == test_y)

    print("acc on the test set: ", test_acc)

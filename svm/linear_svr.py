# -*- coding: utf-8 -*-


import numpy as np
from math import *


class Linear_SVR:
    def __init__(self, w, b, C, tol, lr):
        self.w = w.copy()
        self.b = b
        self.C = C
        self.tol = tol
        self.lr = lr

    def fit(self, X, y, max_iter):
        for i in range(max_iter):
            grad_w = 0
            grad_b = 0
            grad_w += self.C * self.w

            loss_mat = self.w.dot(X.T) + self.b - y
            activation_mat = np.zeros(X.shape[0])
            activation_mat[loss_mat > 0] = 1
            activation_mat[loss_mat < 0] = -1
            loss_mat = np.fabs(loss_mat) - self.tol
            loss_mat[loss_mat < 0] = 0
            activation_mat[loss_mat == 0] = 0

            grad_w += activation_mat.dot(X)
            grad_b += np.sum(activation_mat)

            loss = np.sum(loss_mat)
            if (i+1) % 500 == 0:
                print("iter: {}, loss: {}".format(i+1, loss))

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def predict(self, X):
        return self.w.dot(X.T) + self.b

def R_square(y, y_hat, n, p):
    return 1 - np.sum((y-y_hat)**2) / np.sum((y - np.mean(y))**2) * (n-1) / (n-p-1)


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    dataset = load_diabetes()

    data = dataset['data']
    data = (data - np.min(data, 0)) / (np.max(data, 0) - np.min(data, 0))
    target = dataset['target']

    train_X = data[:350]
    train_y = target[:350]

    valid_X = data[350:400]
    valid_y =target[350:400]

    test_X = data[400:]
    test_y = target[400:]

    w = np.random.randn(10) / sqrt(10)
    b = 0
    lr = 0.0001
    C = 0.001
    tol = 1

    svr = Linear_SVR(w, b, C, tol, lr)
    svr.fit(train_X, train_y, max_iter=80000)

    train_predict = svr.predict(train_X)
    print("Residual matrix on the train set:", train_predict - train_y)
    print("R2 on train set: %f" % R_square(train_y, train_predict, train_X.shape[0], 10))

    valid_predict = svr.predict(valid_X)
    print("Residual matrix on the validation set:", valid_predict - valid_y)
    print("R2 on validation set: %f" % R_square(valid_y, valid_predict, valid_X.shape[0], 10))

    test_predict = svr.predict(test_X)
    print("Residual matrix on the test set:", test_predict - test_y)

    print("R2 on test set: %f" % R_square(test_y, test_predict, test_X.shape[0], 10))


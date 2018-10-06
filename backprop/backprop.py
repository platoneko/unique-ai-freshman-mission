# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from math import *


class FC:
    def __init__(self, W, b, lr, regu_rate):
        self.W = W.copy()
        self.b = b.copy()
        self.lr = lr
        self.regu_rate = regu_rate

    def forward(self, X):
        self.X = X.copy()
        return self.X.dot(self.W) + self.b

    def backprop(self, back_grad):
        self.grad_W = self.X.T.dot(back_grad)
        self.grad_b = np.ones(self.X.shape[0]).dot(back_grad)
        self.grad = back_grad.dot(self.W.T)
        return self.grad

    def update(self):
        self.W -= self.lr * (self.grad_W + self.regu_rate * self.W)
        self.b -= self.lr * self.grad_b


class Relu:
    def forward(self, X):
        self.X = X.copy()
        return np.maximum(X, 0)

    def backprop(self, back_grad):
        grad = back_grad.copy()
        grad[self.X < 0] = 0
        return grad


class SparseSoftmaxCrossEntropy:
    def forward(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
        denom = np.sum(np.exp(self.X), axis=1).reshape([-1, 1])
        self.softmax = np.exp(X) / denom
        cross_entropy = np.mean(-np.log(self.softmax[range(self.X.shape[0]), self.y]))
        return cross_entropy

    def backprop(self):
        m, n = self.X.shape
        activation_mat = np.zeros([m, n])
        activation_mat[range(m), self.y] = 1
        grad = (self.softmax - activation_mat) / m
        return grad


def bp_test():
    CLASS_DICT = {'very_low': 0, 'Low': 1, 'Middle': 2, 'High': 3, 'Very Low': 0}
    train_data = pd.read_excel('./data/data5.xls', sheetname='Training_Data')

    labels = np.array(train_data.iloc[:, -1].map(CLASS_DICT)).astype(np.int)
    data = np.delete(np.array(train_data), -1, 1).astype(np.float)

    valid_X = data[200:]
    valid_y = labels[200:]

    train_X = data[:200]
    train_y = labels[:200]

    W1 = np.random.randn(5, 64) / sqrt(5)
    b1 = np.zeros(64)
    W2 = np.random.randn(64, 32) / sqrt(64)
    b2 = np.zeros(32)
    lr = 0.05
    regu_rate = 0.001
    max_iter = 30000

    fc1 = FC(W1, b1, lr, regu_rate)
    relu1 = Relu()
    fc2 = FC(W2, b2, lr, regu_rate)
    cross_entropy = SparseSoftmaxCrossEntropy()

    for i in range(max_iter):
        h1 = fc1.forward(train_X)
        h2 = relu1.forward(h1)
        h3 = fc2.forward(h2)
        loss = cross_entropy.forward(h3, train_y)
        if (i+1)%500 == 0:
            print("iter: {}, loss：{}".format(i+1, loss))

        grad_h3 = cross_entropy.backprop()
        grad_h2 = fc2.backprop(grad_h3)
        grad_h1 = relu1.backprop(grad_h2)
        grad_X = fc1.backprop(grad_h1)

        fc2.update()
        fc1.update()

    valid_h1 = fc1.forward(valid_X)
    valid_h2 = relu1.forward(valid_h1)
    valid_h3 = fc2.forward(valid_h2)
    valid_predict = np.argmax(valid_h3, 1)

    valid_acc = np.mean(valid_predict == valid_y)

    print('validating acc: ', valid_acc)

    test_data = pd.read_excel('./data/data5.xls', sheetname='Test_Data')

    test_y = np.array(test_data.iloc[:, -1].map(CLASS_DICT)).astype(np.int)
    test_X = np.delete(np.array(test_data), -1, 1).astype(np.float)

    test_h1 = fc1.forward(test_X)
    test_h2 = relu1.forward(test_h1)
    test_h3 = fc2.forward(test_h2)
    test_predict = np.argmax(test_h3, 1)

    acc = np.mean(test_predict == test_y)

    print('test acc: ', acc)



if __name__ == '__main__':
    bp_test()

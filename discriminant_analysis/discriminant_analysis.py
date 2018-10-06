# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from math import *


class QDA:
    def fit(self, X, y):
        m, n = X.shape
        self.miu = np.zeros([2, n])
        self.pi = np.zeros(2)
        self.sigma = [0]*2
        self.sigma_I = [0]*2
        self.det_sigma = [0] * 2
        for i in range(2):
            self.pi[i] = np.sum(y == i) / m
            self.miu[i] = np.mean(X[y == i], 0)
            normal_X = X[y == i] - self.miu[i]
            self.sigma[i] = normal_X.T.dot(normal_X) / (np.sum(y == i) + 1)
            self.sigma_I[i] = np.linalg.inv(self.sigma[i])
            self.det_sigma[i] = np.linalg.det(self.sigma[i])

    def predict(self, X):
        X_minus_miu0 = X - self.miu[0]
        X_minus_miu1 = X - self.miu[1]

        prob_mat = log(self.pi[0]/self.pi[1]) - 0.5 * log(self.det_sigma[0]/self.det_sigma[1]) \
                   - 0.5 * np.sum(X_minus_miu0.dot(self.sigma_I[0]) * X_minus_miu0, 1) + \
                   0.5 * np.sum(X_minus_miu1.dot(self.sigma_I[1]) * X_minus_miu1, 1)

        predict_mat = np.zeros(X.shape[0])
        predict_mat[prob_mat < 0] = 1
        return predict_mat


class LDA:
    def fit(self, X, y):
        m, n = X.shape
        self.miu = np.zeros([2, n])
        self.pi = np.zeros(2)
        self.sigma = np.zeros([n, n])
        for i in range(2):
            self.pi[i] = np.sum(y == i) / m
            self.miu[i] = np.mean(X[y == i], 0)
            normal_X = X[y == i] - self.miu[i]
            self.sigma += normal_X.T.dot(normal_X)
        self.sigma /= m-2
        self.sigma_I = np.linalg.inv(self.sigma)

    def predict(self, X):
        prob_mat = log(self.pi[0]/self.pi[1]) + \
                   np.sum(X.dot(self.sigma_I) * (self.miu[0] - self.miu[1]), 1) - \
                   0.5*np.sum((self.miu[0] + self.miu[1]).dot(self.sigma_I) * (self.miu[0] - self.miu[1]))

        predict_mat = np.zeros(X.shape[0])
        predict_mat[prob_mat < 0] = 1
        return predict_mat


if __name__ == "__main__":
    train_data_set = pd.read_excel('data.xlsx', sheetname='traindata')
    train_labels = np.array(train_data_set.iloc[:, -1]).astype(np.int)
    train_data_ori = np.delete(np.array(train_data_set), -1, 1).astype(np.float)

    #train_data = (train_data_ori - np.mean(train_data_ori, 0)) / np.std(train_data_ori, 0)
    train_data = (train_data_ori - np.min(train_data_ori, 0)) / (np.max(train_data_ori, 0) - np.min(train_data_ori, 0))

    train_X = train_data_ori
    train_y = train_labels

    test_data_set = pd.read_excel('data.xlsx', sheetname='testdata')
    test_labels = np.array(test_data_set.iloc[:, -1]).astype(np.int)
    test_data_ori = np.delete(np.array(test_data_set), -1, 1).astype(np.float)

    #test_data = (test_data_ori - np.mean(test_data_ori, 0)) / np.std(test_data_ori, 0)
    test_data = (test_data_ori - np.min(test_data_ori, 0)) / (np.max(test_data_ori, 0) - np.min(test_data_ori, 0))

    test_X = test_data_ori
    test_y = test_labels

    clf = LDA()
    clf.fit(train_X, train_y)

    test_predict = clf.predict(test_X)
    test_acc = np.mean(test_predict == test_y)
    print("acc on the test set:", test_acc)

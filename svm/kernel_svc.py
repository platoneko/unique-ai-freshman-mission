# -*- coding: utf-8 -*-


import numpy as np
import random
import time
from math import *


class Kernel_SVC:
    def __init__(self, C, kernel, tol=0.001, eps=0.001, max_iter=1000, **kwargs):
        self.C = C
        self.kernel = getattr(self, kernel+'_kernel')
        self.tol = tol
        self.eps = eps
        self.max_iter = max_iter
        self.kwargs = kwargs

    def fit(self, X, y):
        self.X = X
        self.y = y
        m, n = self.X.shape
        self.alpha = np.zeros(m)
        self.b = 0
        self.fX = np.zeros(m)
        if self.kernel == self.linear_kernel:
            self.W = np.zeros([n, 1])
        num_changed = 0
        examine_all = True
        iter_num = 0
        while ((num_changed > 0 or examine_all) and self.max_iter > iter_num):
            iter_num += 1
            num_changed = 0
            if examine_all:
                for i in range(m):
                    num_changed += self.examine_example(i)
            else:
                for i in range(m):
                    if (0 < self.alpha[i] < self.C):
                        num_changed += self.examine_example(i)
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
        print(iter_num)

    def examine_example(self, i2):
        m = self.X.shape[0]
        alpha2 = self.alpha[i2]
        E2 = self.fX[i2] - self.y[i2]
        r2 = E2 * self.y[i2]
        if (r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0):
            candidate = np.nonzero((self.alpha != self.C) & (self.alpha != 0))[0]
            if len(candidate) > 1:
                i1 = self.choice_heuristic(E2)
                if self.take_step(i1, i2, alpha2, E2):
                    return 1
            if len(candidate):
                for i in range(random.randrange(0, len(candidate)), len(candidate)):
                    i1 = candidate[i]
                    if self.take_step(i1, i2, alpha2, E2):
                        return 1
            for i in range(random.randrange(0, m), m):
                i1 = i
                if self.take_step(i1, i2, alpha2, E2):
                    return 1
        return 0

    def take_step(self, i1, i2, alpha2, E2):
        if i1 == i2:
            return 0
        alpha1 = self.alpha[i1]
        E1 = self.fX[i1] - self.y[i1]
        s = self.y[i1] * self.y[i2]
        if self.y[i1] == self.y[i2]:
            L = max(0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)
        else:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        if L == H:
            return 0
        k11 = self.kernel(self.X[i1], self.X[i1])
        k12 = self.kernel(self.X[i1], self.X[i2])
        k22 = self.kernel(self.X[i2], self.X[i2])
        eta = k11 + k22 - 2*k12
        if eta > 0:
            a2 = alpha2 + self.y[i2] * (E1-E2)/eta
            a2 = max(L, min(a2, H))
        else:
            f1 = self.y[i1] * (E1 - self.b) - alpha1 * k11 - s * alpha2 * k12
            f2 = self.y[i2] * (E2 - self.b) - s * alpha1 * k12 - alpha2 * k22
            L1 = alpha1 + s * (alpha2 - L)
            H1 = alpha1 + s * (alpha2 - H)
            L_obj = L1*f1 + L*f2 + 0.5*(L1**2)*k11 + 0.5*(L**2)*k22 + s*L*L1*k12
            H_obj = H1*f1 + H*f2 + 0.5*(H1**2)*k11 + 0.5*(H**2)*k22 + s*H*H1*k12
            if L_obj < H_obj-self.eps:
                a2 = L
            elif L_obj > H_obj+self.eps:
                a2 = H
            else:
                a2 = alpha2
        if fabs(a2 - alpha2) < self.eps * (a2+alpha2+self.eps):
            return 0
        a1 = alpha1 + s*(alpha2-a2)
        b1 = -E1 - self.y[i1] * k11 * (a1 - alpha1) - self.y[i2] * k12 * (a2 - alpha2) + self.b
        b2 = -E2 - self.y[i1] * k12 * (a1 - alpha1) - self.y[i2] * k22 * (a2 - alpha2) + self.b
        self.b = (b1 + b2) / 2
        self.alpha[i1] = np.round(a1, 2)
        self.alpha[i2] = np.round(a2, 2)
        #print(self.alpha)
        #print("b: ", self.b)
        if  self.kernel == self.linear_kernel:
            self.W += (self.y[i1]*(a1 - alpha1)*self.X[i1].reshape([-1, 1]) +
                      self.y[i2]*(a2 - alpha2)*self.X[i2].reshape([-1,1]))
        self.fX = self.infer(self.X)
        return 1

    def choice_heuristic(self, E2):
        return np.argmax(np.fabs(self.fX - self.y - E2))

    def infer(self, X):
        fX = np.zeros(X.shape[0])
        if self.kernel == self.linear_kernel:
            fX = X.dot(self.W).reshape(-1) + self.b
        else:
            for i in range(X.shape[0]):
                fX[i] = np.sum(self.alpha * self.y * self.kernel(X[i], self.X)) + self.b
        return fX

    def predict(self, X):
        fX = self.infer(X)
        return np.sign(fX)

    def rbf_kernel(self, xi, xj):
        gamma = self.kwargs['gamma']
        if xj.ndim == 2:
            return np.exp(-np.sum((xi - xj)**2, 1) * (gamma**2))
        return exp(-np.sum((xi - xj)**2) * (gamma**2))

    def laplace_kernel(self, xi, xj):
        gamma = self.kwargs['gamma']
        if xj.ndim == 2:
            return np.exp(-gamma * np.sum((xi - xj)**2, 1)**0.5)
        return exp(-gamma * np.sum((xi - xj)**2)**0.5)

    def sigmoid_kernel(self, xi, x):
        beta = self.kwargs['beta']
        theta = self.kwargs['theta']
        if x.ndim == 2:
            return np.tanh(beta * xi.dot(x.T) + theta).reshape(-1)
        return tanh(beta * xi.dot(x.reshape([-1, 1])) + theta)

    def poly_kernel(self, xi, xj):
        d = self.kwargs['d']
        if xj.ndim == 2:
            return ((xi.dot(xj.T))**d).reshape(-1)
        return xi.dot(xj.reshape([-1, 1]))**d

    def linear_kernel(self, xi, xj):
        return xi.dot(xj.reshape([-1, 1]))


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    dataset = load_breast_cancer()
    data = dataset['data']
    labels = dataset['target']
    labels[labels == 0] = -1

    data = (data - np.mean(data, 0)) / np.std(data, 0)

    train_x = data[:400]
    train_y = labels[:400]
    valid_x = data[400:500]
    valid_y = labels[400:500]
    test_x = data[500:]
    test_y = labels[500:]

    t1 = time.time()
    clf = Kernel_SVC(C=1000, kernel='rbf', gamma=0.001)
    clf.fit(train_x, train_y)

    train_predict = clf.predict(train_x)
    train_acc = np.mean(train_predict == train_y)
    print("acc on training set:", train_acc)

    predict = clf.predict(valid_x)
    acc = np.mean(predict == valid_y)
    print("acc on validation set:", acc)

    test_predict = clf.predict(test_x)
    print(test_predict)
    print(test_y)
    test_acc = np.mean(test_predict == test_y)
    print("acc on test set:", test_acc)

    t2 = time.time()
    print("time cost: %fs"% (t2 - t1))

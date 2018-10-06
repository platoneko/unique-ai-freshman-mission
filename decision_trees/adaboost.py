# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from math import *


def get_features_bag(features):
    features_bag = []
    for i in range(features.shape[1]):
        features_bag.append(set(features[:,i]))
    return features_bag


def split(features, feature, value, op):
    m, n = features.shape
    estimation = np.ones(m)
    if op == 'less':
        for i in range(m):
            if features[i, feature] <= value:
                estimation[i] = -1
    else:
        for i in range(m):
            if features[i, feature] > value:
                estimation[i] = -1
    return estimation


def make_tree(features, labels, weights, features_bag):
    m, n = features.shape
    estimation = np.ones(m)
    min_error_rate = np.inf
    for feature in range(n):
        for value in features_bag[feature]:
            for op in ['less', 'greater']:
                estimation = split(features, feature, value, op)
                error_rate = np.sum(weights * (estimation != labels))
                if error_rate < min_error_rate:
                    min_error_rate = error_rate
                    best_feature = feature
                    best_value = value
                    best_op = op
                    best_estimation = estimation
    tree = {'feature': best_feature, 'value': best_value, 'op': best_op}
    return tree, min_error_rate, best_estimation


def adaboost_train(features, labels, tree_num=40):
    trees = []
    alpha_list = []
    m, n = features.shape
    weights = np.zeros(m) + 1/m
    features_bag = get_features_bag(features)
    for i in range(tree_num):
        tree, weight_error_rate, estimation = make_tree(features, labels, weights, features_bag)
        if weight_error_rate >= 0.5:
            break
        alpha = 0.5 * log((1 - weight_error_rate)/max(weight_error_rate, 1e-6))
        alpha_list.append(alpha)
        trees.append(tree)
        expon = -alpha * labels * estimation
        weights = weights * np.exp(expon)
        weights = weights / np.sum(weights)
    return trees, alpha_list


def classify(trees, alpha_list, input_feature):
    total_class = 0
    for i in range(len(trees)):
        if trees[i]['op'] == 'less':
            if input_feature[trees[i]['feature']] <= trees[i]['value']:
                class_ = -1
            else:
                class_ = 1
        else:
            if input_feature[trees[i]['feature']] > trees[i]['value']:
                class_ = -1
            else:
                class_ = 1
        total_class += alpha_list[i] * class_
    return -1 if total_class < 0 else 1


if __name__ == '__main__':

    def is_string(input):
        result = np.zeros(len(input))
        for i in range(len(input)):
            if isinstance(input[i], str):
                result[i] = 1
        return  result


    train_data = pd.read_csv('./data/train.csv')
    train_features = np.zeros([len(train_data), 7])
    train_labels = np.zeros(len(train_data))
    for i in range(len(train_data)):
        train_features[i, 0] = train_data['Pclass'][i]

        if train_data['Sex'][i] == 'male':
            train_features[i, 1] = 1

        train_features[i, 2] = train_data['SibSp'][i]

        train_features[i, 3] = train_data['Parch'][i]

        if train_data['Embarked'][i] == 'S':
            train_features[i, 4] = 1
        elif train_data['Embarked'][i] == 'C':
            train_features[i, 4] = 2
        elif train_data['Embarked'][i] == 'Q':
            train_features[i, 4] = 3
        else:
            train_features[i, 4] = 0

        if np.isnan(train_data['Age'][i]):
            train_features[i, 5] = 0
        else:
            train_features[i, 5] = train_data['Age'][i] // 10 + 1

        if train_data['Survived'][i] == 1:
            train_labels[i] = 1
        else:
            train_labels[i] = -1

    train_features[:, 6] = is_string(train_data['Cabin'])

    trees, alpha_list = adaboost_train(train_features, train_labels, tree_num=2)
    print(trees)

    test_data = pd.read_csv('./data/test.csv')
    test_label_data = pd.read_csv('./data/gender_submission.csv')
    test_features = np.zeros([len(test_data), 7])
    test_labels = np.zeros(len(test_data))
    for i in range(len(test_data)):
        test_features[i, 0] = test_data['Pclass'][i]

        if test_data['Sex'][i] == 'male':
            test_features[i, 1] = 1

        test_features[i, 2] = test_data['SibSp'][i]

        test_features[i, 3] = test_data['Parch'][i]

        if test_data['Embarked'][i] == 'S':
            test_features[i, 4] = 1
        elif test_data['Embarked'][i] == 'C':
            test_features[i, 4] = 2
        elif test_data['Embarked'][i] == 'Q':
            test_features[i, 4] = 3
        else:
            test_features[i, 4] = 0

        if np.isnan(test_data['Age'][i]):
            test_features[i, 5] = 0
        else:
            test_features[i, 5] = test_data['Age'][i] // 10 + 1

        if test_label_data['Survived'][i] == 1:
            test_labels[i] = 1
        else:
            test_labels[i] = -1

    test_features[:, 6] = is_string(test_data['Cabin'])

    correct = 0
    for i in range(len(test_data)):
        if classify(trees, alpha_list, test_features[i]) == test_labels[i]:
            correct += 1
    acc = correct / len(test_data)
    print('acc = {}'.format(acc))
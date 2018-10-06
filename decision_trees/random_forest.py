# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import random

import cart


def generate_index(total, sample_num):
    index_set = set()
    while len(index_set) < sample_num:
        index = random.randrange(0, total)
        if index not in index_set:
            index_set.add(index)
    return list(index_set)


def make_forest(features, labels, feature_num, sample_num, tree_num):
    m, n = features.shape
    forest = []
    for i in range(tree_num):
        sample_index = generate_index(m, sample_num)
        feature_index = generate_index(n, feature_num)
        tree = cart.make_tree(features, labels, feature_index, sample_index, leaf_labels_num=20, accepted_gini=0.4)
        forest.append(tree)
    return forest


def classify(forest, input_feature):
    counter = dict()
    for tree in forest:
        label = cart.classify(tree, input_feature)
        if label not in counter:
            counter[label] = 0
        counter[label] += 1
    return max(counter.items(), key=lambda x: x[1])[0]


if __name__ == '__main__':

    def is_string(input):
        result = np.zeros(len(input))
        for i in range(len(input)):
            if isinstance(input[i], str):
                result[i] = 1
        return  result


    train_data = pd.read_csv('/data/train.csv')
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

    train_features[:, 6] = is_string(train_data['Cabin'])

    train_labels[:] = np.array(train_data['Survived'])

    forest = make_forest(train_features, train_labels, feature_num=4, sample_num=100, tree_num=40)

    test_data = pd.read_csv('/data/test.csv')
    test_label_data = pd.read_csv('/data/gender_submission.csv')
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
    test_features[:, 6] = is_string(test_data['Cabin'])
    test_labels[:] = np.array(test_label_data['Survived'])

    correct = 0
    for i in range(len(test_data)):
        if classify(forest, test_features[i]) == test_labels[i]:
            correct += 1
    acc = correct / len(test_data)
    print('acc = {}'.format(acc))
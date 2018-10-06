# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd


def get_features_bag(features, feature_list, index_list):
    features_bag = dict()
    stop_recurse = True
    for i in feature_list:
        features_bag[i] = set(features[index_list][:,i])
        if len(features_bag[i]) > 1:
            stop_recurse = False
    return features_bag, stop_recurse


def majority_label(labels, index_list):
    labels_dict = dict()
    for index in index_list:
        if labels[index] not in labels_dict:
            labels_dict[labels[index]] = 0
        labels_dict[labels[index]] += 1
    return max(labels_dict.items(), key=lambda x: x[1])[0]


def split(features, index_list, feature, value):
    left_index_list = []
    right_index_list = []
    for index in index_list:
        if features[index][feature] < value:
            left_index_list.append(index)
        else:
            right_index_list.append(index)
    return left_index_list, right_index_list


def get_gini(labels, index_list):
    gini = 1
    labels_dict = dict()
    for label in labels[index_list]:
        if label not in labels_dict:
            labels_dict[label] = 0
        labels_dict[label] += 1
    for label in labels_dict:
        gini -= (labels_dict[label]/len(index_list)) ** 2
    return gini


def get_best_split(features, labels, feature_list, index_list, features_bag):
    min_gini = 1
    best_feature = feature_list[-1]
    best_value = list(features_bag[best_feature])[-1]
    for feature in feature_list:
        for value in features_bag[feature]:
            left_index_list, right_index_list = split(features, index_list, feature, value)
            left_prob = len(left_index_list) / len(index_list)
            right_prob = len(right_index_list) / len(index_list)
            gini = left_prob * get_gini(labels, left_index_list) + right_prob * get_gini(labels, right_index_list)
            if gini < min_gini:
                min_gini = gini
                best_feature = feature
                best_value = value
    return best_feature, best_value


def make_tree(features, labels, feature_list=None, index_list=None, leaf_labels_num=6, accepted_gini=0.1):
    if feature_list is None:
        feature_list = list(range(features.shape[1]))
    if index_list is None:
        index_list = list(range(len(labels)))
    if len(set(labels[index_list])) == 1:
        return labels[index_list][0]
    if len(index_list) <= leaf_labels_num or get_gini(labels, index_list) < accepted_gini:
        return majority_label(labels, index_list)

    features_bag, stop_recurse = get_features_bag(features, feature_list, index_list)
    if stop_recurse:
        return majority_label(labels, index_list)

    best_feature, best_value = get_best_split(features, labels, feature_list, index_list, features_bag)

    tree = {best_feature: {best_value: {}}}
    left_index_list, right_index_list = split(features, index_list, best_feature, best_value)
    tree[best_feature][best_value]['left'] = make_tree(features, labels, feature_list, left_index_list,
                                                       leaf_labels_num, accepted_gini)
    tree[best_feature][best_value]['right'] = make_tree(features, labels, feature_list, right_index_list,
                                                        leaf_labels_num, accepted_gini)

    return tree


def classify(tree, input_feature):
    if not isinstance(tree, dict): #  Maybe it is a label now, for instance data has the same features
        return tree
    feature = list(tree.keys())[0]
    value = list(tree[feature].keys())[0]
    if input_feature[feature] < value:
        sub_tree = tree[feature][value]['left']
    else:
        sub_tree = tree[feature][value]['right']
    if isinstance(sub_tree, dict):
        return classify(sub_tree, input_feature)
    return sub_tree


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

    train_features[:, 6] = is_string(train_data['Cabin'])

    train_labels[:] = np.array(train_data['Survived'])

    tree = make_tree(train_features, train_labels, leaf_labels_num=30, accepted_gini=0.4)

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
    test_features[:, 6] = is_string(test_data['Cabin'])
    test_labels[:] = np.array(test_label_data['Survived'])

    correct = 0
    for i in range(len(test_data)):
        if classify(tree, test_features[i]) == test_labels[i]:
            correct += 1
    acc = correct / len(test_data)
    print('acc = {}'.format(acc))
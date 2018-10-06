# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from math import *


def load_watermelon_dataset():
    """
    features index dict
    {0: "色泽", 1: "根蒂", 2: "敲声", 3: "纹理", 4: "脐部", 5: "触感", 6: "密度", 7: "含糖率"}
    features value dict
    {"色泽": {0: "浅白", 1: "青绿", 2: "乌黑"},
     "根蒂": {0: "蜷缩", 1: "稍蜷", 2: "硬挺"},
     "敲声": {0: "清脆", 1: "沉闷", 2: "浊响"},
     "纹理": {0: "清晰", 1: "稍糊", 2: "模糊"},
     "脐部": {0: "平坦", 1: "稍凹", 2: "凹陷"},
     "触感": {0: "硬滑", 1: "软粘"}}
    """
    x = np.array([[1, 0, 2, 0, 2, 0, 0.697, 0.460],
                  [2, 0, 1, 0, 2, 0, 0.774, 0.376],
                  [2, 0, 2, 0, 2, 0, 0.634, 0.264],
                  [1, 0, 1, 0, 2, 0, 0.608, 0.318],
                  [0, 0, 2, 0, 2, 0, 0.556, 0.215],
                  [1, 1, 2, 0, 1, 1, 0.403, 0.237],
                  [2, 1, 2, 1, 1, 1, 0.481, 0.149],
                  [2, 1, 2, 0, 1, 0, 0.437, 0.211],
                  [2, 1, 1, 1, 1, 0, 0.666, 0.091],
                  [1, 2, 0, 0, 0, 1, 0.243, 0.267],
                  [0, 2, 0, 2, 0, 0, 0.245, 0.057],
                  [0, 0, 2, 2, 0, 1, 0.343, 0.099],
                  [1, 1, 2, 1, 2, 0, 0.639, 0.161],
                  [0, 1, 1, 1, 2, 0, 0.657, 0.198],
                  [2, 1, 2, 0, 1, 1, 0.360, 0.370],
                  [0, 0, 2, 2, 0, 0, 0.593, 0.042],
                  [1, 0, 1, 1, 1, 0, 0.719, 0.103]])
    y = np.array([1]*8 + [-1]*9)
    return x, y


def get_features_bag(x, index_list):
    features_bag = dict()
    stop_recurse = True
    for i in range(x.shape[1]):
        features_bag[i] = set(x[index_list][:,i])
        if len(features_bag[i]) > 1:
            stop_recurse = False
    return features_bag, stop_recurse


def split(x, index_list, feature, value):
    left_index_list = []
    right_index_list = []
    for index in index_list:
        if x[index][feature] < value:
            left_index_list.append(index)
        else:
            right_index_list.append(index)
    return left_index_list, right_index_list


def get_best_split(x, y, weights, index_list, features_bag):
    min_gini = 1
    best_feature = 0
    best_value = list(features_bag[best_feature])[-1]
    denom = np.sum(weights[index_list])
    for feature in range(x.shape[1]):
        for value in features_bag[feature]:
            left_index_list, right_index_list = split(x, index_list, feature, value)
            left_prob = np.sum(weights[left_index_list]) / denom
            right_prob = np.sum(weights[right_index_list]) / denom
            gini = left_prob * get_gini(y, weights, left_index_list) + right_prob * get_gini(y, weights, right_index_list)
            if gini < min_gini:
                min_gini = gini
                best_feature = feature
                best_value = value
    return best_feature, best_value


def get_gini(y, weights, index_list):
    gini = 1
    labels_dict = dict()
    for label in y[index_list]:
        if label not in labels_dict:
            labels_dict[label] = 0
        labels_dict[label] += weights[label]
    for label in labels_dict:
        gini -= (labels_dict[label] / np.sum(weights[index_list])) ** 2
    return gini


def majority_label(labels, index_list):
    labels_dict = dict()
    for index in index_list:
        if labels[index] not in labels_dict:
            labels_dict[labels[index]] = 0
        labels_dict[labels[index]] += 1
    return max(labels_dict.items(), key=lambda x: x[1])[0]


def make_tree(x, y, weights, index_list=None, leaf_labels_num=4, accepted_gini=0.4):
    if index_list is None:
        index_list = list(range(len(y)))
    if len(set(y[index_list])) == 1:
        return y[index_list][0]
    if len(index_list) <= leaf_labels_num or get_gini(y, weights, index_list) < accepted_gini:
        return majority_label(y, index_list)

    features_bag, stop_recurse = get_features_bag(x, index_list)
    if stop_recurse:
        return majority_label(y, index_list)

    best_feature, best_value = get_best_split(x, y, weights, index_list, features_bag)

    tree = {best_feature: {best_value: {}}}
    left_index_list, right_index_list = split(x, index_list, best_feature, best_value)
    tree[best_feature][best_value]['left'] = make_tree(x, y, weights, left_index_list,
                                                       leaf_labels_num, accepted_gini)
    tree[best_feature][best_value]['right'] = make_tree(x, y, weights, right_index_list,
                                                        leaf_labels_num, accepted_gini)

    return tree


def adaboost_train(x, y, tree_num=10):
    trees = []
    alpha_list = []
    m, n = x.shape
    weights = np.zeros(m) + 1/m
    for i in range(tree_num):
        tree = make_tree(x, y, weights)
        estimation = np.ones(m)
        for j in range(m):
            estimation[j] = classify(tree, x[j])
        weight_error_rate = np.sum(weights * (estimation != y))
        if weight_error_rate >= 0.5:
            break
        alpha = 0.5 * log((1 - weight_error_rate)/max(weight_error_rate, 1e-6))
        alpha_list.append(alpha)
        trees.append(tree)
        expon = -alpha * y * estimation
        weights = weights * np.exp(expon)
        weights = weights / np.sum(weights)
    return trees, alpha_list


def classify(tree, input_x):
    if not isinstance(tree, dict): #  Maybe it is a label now, for instance data has the same features
        return tree
    feature = list(tree.keys())[0]
    value = list(tree[feature].keys())[0]
    if input_x[feature] < value:
        sub_tree = tree[feature][value]['left']
    else:
        sub_tree = tree[feature][value]['right']
    if isinstance(sub_tree, dict):
        return classify(sub_tree, input_x)
    return sub_tree


def adaboost_classify(trees, alpha_list, input_x):
    total_result = 0
    for i in range(len(trees)):
        result = classify(trees[i], input_x)
        total_result += alpha_list[i] * result
    sign_result = -1 if total_result < 0 else 1
    return sign_result, total_result


def loo(x, y):
    result_array = np.zeros(len(x))
    prob_array = np.zeros(len(x))
    for i in range(len(x)):
        index_list = set(range(len(x)))
        index_list.remove(i)
        index_list = list(index_list)
        trees, alpha_list = adaboost_train(x[index_list], y[index_list])
        result_array[i], prob_array[i] = adaboost_classify(trees, alpha_list, x[i])
    print("classification result:", result_array)
    acc = np.sum(y == result_array) / len(x)
    print("accuracy:", acc)

    roc(prob_array, y)
    pr(prob_array, y)


def roc(prob_array, y):
    point = (0.0, 0.0)
    m = len(y)
    pos_num = np.sum(y == 1)
    y_step = 1 / pos_num
    x_step = 1 / (m - pos_num)
    sorted_list = prob_array.argsort()
    y_sum = 0
    for index in reversed(sorted_list.tolist()):
        if y[index] == 1:
            delta_x, delta_y = 0, y_step
        else:
            delta_x, delta_y = x_step, 0
            y_sum += point[1]
        plt.plot([point[0], point[0] + delta_x], [point[1], point[1] + delta_y], c='b')
        point = (point[0] + delta_x, point[1] + delta_y)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('The AUC is {}'.format(y_sum * x_step))
    plt.show()
    plt.close()


def pr(prob_array, y):
    m = len(prob_array)
    sorted_index = prob_array.argsort().tolist()
    sorted_index.reverse()
    tp = 0
    num1 = np.sum(y == 1)
    precision_list = [1.0]
    recall_list = [0.0]
    for i in range(m):
        if y[sorted_index[i]] == 1:
            tp += 1
        precision = tp / (i+1)
        recall = tp / num1
        precision_list.append(precision)
        recall_list.append(recall)
    precision_list.append(0.0)
    recall_list.append(1.0)
    plt.plot(recall_list, precision_list, c='b')
    plt.xlabel('recall rate')
    plt.ylabel('precision rate')
    plt.show()
    plt.close()


if __name__ == "__main__":
    x, y = load_watermelon_dataset()
    loo(x, y)
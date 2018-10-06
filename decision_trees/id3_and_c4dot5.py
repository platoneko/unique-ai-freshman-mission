# -*- coding: utf-8 -*-


import numpy as np
import random
import matplotlib.pyplot as plt
from math import *


def get_features_bag(x, index_list):
    features_bag = []
    for i in range(x.shape[1]):
        features_bag.append(set(x[index_list][:,i]))
    return features_bag


def split(x, index_list, feature, value):
    sub_index_list = []
    for index in index_list:
        if x[index, feature] == value:
            sub_index_list.append(index)
    return sub_index_list


def get_entropy(y, index_list):
    labels_dict = dict()
    for index in index_list:
        if y[index] not in labels_dict:
            labels_dict[y[index]] = 0
        labels_dict[y[index]] += 1
    entropy = 0
    for label in labels_dict:
        entropy -= labels_dict[label]/len(index_list) * log(labels_dict[label] / len(index_list), 2)
    return entropy


def majority_label(y, index_list):
    labels_dict = dict()
    for index in index_list:
        if y[index] not in labels_dict:
            labels_dict[y[index]] = 0
        labels_dict[y[index]] += 1
    return max(labels_dict.items(), key=lambda x: x[1])[0]


def ID3(x, y, index_list, features_set, features_bag):
    min_entropy = np.inf
    for feature in features_set:
        entropy = 0
        for value in features_bag[feature]:
            sub_index_list = split(x, index_list, feature, value)
            prob = len(sub_index_list) / len(index_list)
            entropy += prob * get_entropy(y, sub_index_list)
        if entropy < min_entropy:
            min_entropy = entropy
            best_feature = feature
    return best_feature


def C4dot5(x, y, index_list, features_set, features_bag):
    entropy = get_entropy(y, index_list)
    max_info_gain_rate = 0
    best_feature = list(features_set)[-1]
    for feature in features_set:
        conditional_entropy = 0
        feature_entropy = 0
        for value in features_bag[feature]:
            sub_index_list = split(x, index_list, feature, value)
            prob = len(sub_index_list) / len(index_list)
            conditional_entropy += prob * get_entropy(y, sub_index_list)
            feature_entropy -= prob * log(prob, 2)
        info_gain_rate = (entropy - conditional_entropy)/max(feature_entropy, 1e-6)

        if info_gain_rate > max_info_gain_rate:
            max_info_gain_rate = info_gain_rate
            best_feature = feature
    return best_feature


def make_tree(x, y, index_list=None, features_set=None, algorithm=C4dot5):
    if features_set is None:
        features_set = set(range(x.shape[1]))
    if index_list is None:
        index_list = list(range(len(y)))

    if len(set(y[index_list])) == 1:
        return y[index_list][0]

    majority = majority_label(y, index_list)

    if len(features_set) == 0:
        return majority

    features_bag = get_features_bag(x, index_list)

    best_feature = algorithm(x, y, index_list, features_set, features_bag)

    features_set.remove(best_feature)

    tree = {best_feature: {}}

    for value in features_bag[best_feature]:
        tree[best_feature][value] = make_tree(x, y, split(x, index_list, best_feature, value),
                                              features_set.copy())

    return tree


def make_pre_pruning_tree(x, y, eval_x, eval_y, index_list=None, eval_index_list=None,
                          features_set=None, algorithm=C4dot5):
    if features_set is None:
        features_set = set(range(x.shape[1]))
    if index_list is None:
        index_list = list(range(len(y)))
    if eval_index_list is None:
        eval_index_list = list(range(len(eval_y)))

    if len(set(y[index_list])) == 1:
        return y[index_list][0]

    majority = majority_label(y, index_list)

    if len(features_set) == 0:
        return majority

    if len(eval_index_list):
        features_bag = get_features_bag(x, index_list)

        best_feature = algorithm(x, y, index_list, features_set, features_bag)

        not_split_acc = np.sum(eval_y[eval_index_list] == majority) / len(eval_index_list)
        print('not split acc:', not_split_acc)
        sub_index_list = dict()
        sub_eval_index_list = dict()
        correct = 0
        for value in features_bag[best_feature]:
            sub_index_list[value] = split(x, index_list, best_feature, value)
            sub_eval_index_list[value] = np.nonzero(eval_x[eval_index_list][:, best_feature] == value)[0]
            sub_majority = majority_label(y, sub_index_list[value])
            correct += np.sum(eval_y[sub_eval_index_list[value]] == sub_majority)

        split_acc = correct / len(eval_y)
        print('split acc:', split_acc)
        if not_split_acc > split_acc:
            return majority

        features_set.remove(best_feature)

        tree = {best_feature: {}}

        for value in features_bag[best_feature]:
            tree[best_feature][value] = make_pre_pruning_tree(x, y, eval_x, eval_y,
                                                              sub_index_list[value],
                                                              sub_eval_index_list[value], features_set.copy())

        return tree

    else:
        return majority


def pre_pruning_tree_classify(tree, input_x):
    feature = list(tree.keys())[0]
    try:
        sub_tree = tree[feature][input_x[feature]]
        if isinstance(sub_tree, dict):
            return pre_pruning_tree_classify(sub_tree, input_x)
        return sub_tree
    except KeyError:
        print('The feature value is not in this sub tree!')
        return 'unacc'


def make_tree_with_acc(x, y, eval_x, eval_y, index_list=None, eval_index_list=None, features_set=None, algorithm=C4dot5):
    if features_set is None:
        features_set = set(range(x.shape[1]))
    if index_list is None:
        index_list = list(range(len(y)))
    if eval_index_list is None:
        eval_index_list = list(range(len(eval_y)))

    if len(set(y[index_list])) == 1:
        return {'label': y[index_list][0]}

    majority = majority_label(y, index_list)

    if len(features_set) == 0:
        return {'label': majority}

    if len(eval_index_list):
        features_bag = get_features_bag(x, index_list)

        best_feature = algorithm(x, y, index_list, features_set, features_bag)
        features_set.remove(best_feature)

        not_split_acc = np.sum(eval_y[eval_index_list] == majority) / len(eval_index_list)

        sub_index_list = dict()
        sub_eval_index_list = dict()
        correct = 0

        for value in features_bag[best_feature]:
            sub_index_list[value] = split(x, index_list, best_feature, value)
            sub_eval_index_list[value] = np.nonzero(eval_x[eval_index_list][:, best_feature] == value)[0]
            sub_majority = majority_label(y, sub_index_list[value])
            correct += np.sum(eval_y[sub_eval_index_list[value]] == sub_majority)

        split_acc = correct / len(eval_index_list)

        tree = {'label': majority, 'pruning': not_split_acc > split_acc, 'feature': {best_feature: {}}}

        for value in features_bag[best_feature]:
            tree['feature'][best_feature][value] = make_tree_with_acc(x, y, eval_x, eval_y, sub_index_list[value],
                                                              sub_eval_index_list[value], features_set.copy())

        return tree

    else:
        return {'label': majority}


def pruning_operation(tree):
    possible_pruning = True
    for feature in tree['feature']:
        for value in tree['feature'][feature]:
            if 'feature' in tree['feature'][feature][value]:
                possible_pruning = False

    pruning = False
    if possible_pruning:
        if tree['pruning']:
            del tree['feature']
            pruning = True
    else:
        for feature in tree['feature']:
            for value in tree['feature'][feature]:
                if 'feature' in tree['feature'][feature][value]:
                    sub_pruning = pruning_operation(tree['feature'][feature][value])
                    if sub_pruning:
                        pruning = True

    return pruning


def make_post_pruning_tree(x, y, eval_x, eval_y, algorithm=C4dot5):
    tree = make_tree_with_acc(x, y, eval_x, eval_y, algorithm=algorithm)
    pruning = True
    while pruning:
        pruning = pruning_operation(tree)
    return tree


def coding_the_leaves(tree, k=0):
    if 'feature' not in tree:
        tree['code'] = k
        k += 1
        return k
    for feature in tree['feature']:
        for value in tree['feature'][feature]:
            k = coding_the_leaves(tree['feature'][feature][value], k)
    return k


def post_pruning_tree_classify(tree, input_x):
    if 'feature' not in tree:
        return tree['label']
    try:
        for feature in tree['feature']:
            return  post_pruning_tree_classify(tree['feature'][feature][input_x[feature]], input_x)
    except KeyError:
        print("The feature value is not in the sub tree!")
        return 'unacc'


def get_leaf_code(tree, input_x):
    if 'feature' not in tree:
        return tree['code']
    for feature in tree['feature']:
        return get_leaf_code(tree['feature'][feature][input_x[feature]], input_x)


def k_folds_v(x, y, k):
    m = len(x)
    index_list = list(range(m))
    random.shuffle(index_list)
    test_num = m // k
    for i in range(k):
        test_head = i * test_num
        test_tail = test_head + test_num
        for j in range(0, i):
            eval_head = j * test_num
            eval_tail = eval_head + test_num

            train_x = x[index_list[:eval_head] + index_list[eval_tail:test_head] + index_list[test_tail:]]
            train_y = y[index_list[:eval_head] + index_list[eval_tail:test_head] + index_list[test_tail:]]

            eval_x = x[eval_head:eval_tail]
            eval_y = y[eval_head:eval_tail]

            tree = make_post_pruning_tree(train_x, train_y, eval_x, eval_y)

            correct = 0
            for l in range(test_head, test_tail):
                if post_pruning_tree_classify(tree, x[l]) == y[l]:
                    correct += 1
            acc = correct / test_num
            print("post pruning tree i: {}, j: {}, acc: {} ".format(i, j, acc))

        for j in range(i+1, k):
            eval_head = j * test_num
            eval_tail = eval_head + test_num

            train_x = x[index_list[:test_head] + index_list[test_tail:eval_head] + index_list[eval_tail:]]
            train_y = y[index_list[:test_head] + index_list[test_tail:eval_head] + index_list[eval_tail:]]

            eval_x = x[eval_head:eval_tail]
            eval_y = y[eval_head:eval_tail]

            tree = make_post_pruning_tree(train_x, train_y, eval_x, eval_y)

            correct = 0
            for l in range(test_head, test_tail):
                if post_pruning_tree_classify(tree, x[l]) == y[l]:
                    correct += 1
            acc = correct / test_num
            print("post pruning tree i: {}, j: {}, acc: {} ".format(i, j, acc))
    get_figures(train_x, train_y, tree)


def no_pruning_tree_k_folds_v(x, y, k):
    m = len(x)
    index_list = list(range(m))
    random.shuffle(index_list)
    test_num = m // k
    for i in range(k):
        test_head = i * test_num
        test_tail = test_head + test_num

        train_x = x[index_list[:test_head] + index_list[test_tail:]]
        train_y = y[index_list[:test_head] + index_list[test_tail:]]

        tree = make_tree(train_x, train_y, )

        correct = 0
        for j in range(test_head, test_tail):
            if pre_pruning_tree_classify(tree, x[j]) == y[j]:
                correct += 1
        acc = correct / test_num
        print("no pruning tree i: {}, acc: {}".format(i, acc))


def get_label_roc(m, leaves_dict, label):
    k = len(leaves_dict)
    pos_acc_array = np.zeros(k)
    pos_num = 0
    for i in range(k):
        pos_acc_array[i] = leaves_dict[i][label] / leaves_dict[i]['total']
        pos_num += leaves_dict[i][label]
    neg_num = m - pos_num
    tp = pos_num
    fp = neg_num
    sorted_list = pos_acc_array.argsort()
    x_axis = [1.0]
    y_axis = [1.0]
    for i in sorted_list:
        tp -= leaves_dict[i][label]
        fp -= leaves_dict[i]['total'] - leaves_dict[i][label]
        x_axis.append(fp / neg_num)
        y_axis.append(tp / pos_num)
    auc = 0
    for i in range(len(x_axis)-1):
        auc += (y_axis[i] + y_axis[i+1]) * (x_axis[i] - x_axis[i+1]) * 0.5

    plt.plot(x_axis, y_axis)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(label + ' as positive class' + '   AUC: '+ str(auc))
    plt.show()
    plt.close()


def get_label_pr(m, leaves_dict, label):
    k = len(leaves_dict)
    pos_acc_array = np.zeros(k)
    pos_num = 0
    for i in range(k):
        pos_acc_array[i] = leaves_dict[i][label] / leaves_dict[i]['total']
        pos_num += leaves_dict[i][label]
    neg_num = m - pos_num
    tp = pos_num
    fp = neg_num
    sorted_list = pos_acc_array.argsort()
    x_axis = [1.0]
    y_axis = [0.0]
    for i in sorted_list[:-1]:
        tp -= leaves_dict[i][label]
        fp -= (leaves_dict[i]['total'] - leaves_dict[i][label])
        x_axis.append(tp / pos_num)
        y_axis.append(tp / (tp + fp))
    x_axis.append(0.0)
    y_axis.append(1.0)
    plt.plot(x_axis, y_axis)
    plt.xlabel('Recall Rate')
    plt.ylabel('Precision Rate')
    plt.title(label + ' as positive class')
    plt.show()
    plt.close()


def get_figures(x, y, tree):
    m, n = x.shape
    k = coding_the_leaves(tree)
    leaves_dict = [{'unacc': 0, 'acc': 0, 'good': 0, 'vgood': 0, 'total': 0} for i in range(k)]
    for i in range(m):
        code = get_leaf_code(tree, x[i])
        leaves_dict[code][y[i]] += 1
        leaves_dict[code]['total'] += 1
    get_label_roc(m, leaves_dict, 'unacc')
    get_label_roc(m, leaves_dict, 'acc')
    get_label_roc(m, leaves_dict, 'good')
    get_label_roc(m, leaves_dict, 'vgood')
    get_label_pr(m, leaves_dict, 'unacc')
    get_label_pr(m, leaves_dict, 'acc')
    get_label_pr(m, leaves_dict, 'good')
    get_label_pr(m, leaves_dict, 'vgood')


if __name__ == '__main__':
    train_x, train_y = [], []
    with open('/data/traindata.txt') as f:
        for line in f.readlines()[1:]:
            split_line = line.strip().split(' ')
            train_x.append(split_line[:-1])
            train_y.append(split_line[-1])
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    tree = make_post_pruning_tree(train_x[:1100], train_y[:1100], train_x[1100:], train_y[1100:])

    test_x = []
    with open('/data/testdata.txt') as f:
        for line in f.readlines()[1:]:
            split_line = line.strip().split(' ')
            test_x.append(split_line)
    test_x = np.array(test_x)
    test_y = np.zeros(len(test_x), dtype='<U11')
    for i in range(len(test_x)):
        test_y[i] = post_pruning_tree_classify(tree, test_x[i])
    print(test_y)
    np.save('test_y', test_y)
# -*- coding: utf-8 -*-


import numpy as np
import random
import matplotlib.pyplot as plt
from math import *


COLORS = ['yellow', 'orange', 'blue', 'grey', 'purple']


def euclid_dist2(a, b):
    return np.sum((a - b) ** 2)


def cos_dist(a, b, sigma=50):
    return exp(- euclid_dist2(a, b) / (2 * sigma**2))


def normalization(input_x):
    max_value = np.max(input_x, 0)
    min_value = np.min(input_x, 0)
    range_value = max_value - min_value
    output_x = (input_x - min_value)/(range_value + 1e-7)
    return output_x


def get_avg(dataset, cluster, dist_metric):
    total_dist = 0
    for i in range(len(cluster)):
        for j in range(i, len(cluster)):
            total_dist += dist_metric(dataset[cluster[i]], dataset[cluster[j]]) ** 0.5
    avg = 2 * total_dist / max((len(cluster) * (len(cluster)-1)), 1)
    return avg


def get_dbi(avg, centres, dist_metric):
    dbi = 0
    for i in range(len(centres)):
        max_value = 0
        for j in range(len(centres)):
            if j == i:
                continue
            value = (avg[i] + avg[j]) / dist_metric(centres[i], centres[j])**0.5
            if value > max_value:
                max_value = value
        dbi += max_value
    return dbi / len(centres)


def k_means(dataset, k, centres=None, max_iter=500, dist_metric=cos_dist):
    m, n = dataset.shape

    if centres is None:
        centres = dataset[random.sample(list(range(m)), k)]
 #  initial_centres = centres.copy()
 #  print("the initial centres are:", centres)

    result = np.array([0] * m)
    for l in range(max_iter):
        clusters = [[] for i in range(k)]
        for i in range(m):
            min_dist = np.inf
            for j in range(k):
                dist = dist_metric(dataset[i], centres[j])
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = j
            clusters[best_cluster].append(i)
            result[i] = best_cluster
        change = False
        for i in range(k):
            new_centre = np.mean(dataset[clusters[i]], 0)
            if dist_metric(new_centre, centres[i]) > 1e-9:
                centres[i] = new_centre
                change = True
        if change == False:
            break

    avg = []
    for i in range(k):
        avg.append(get_avg(dataset, clusters[i], dist_metric=dist_metric))

    dbi = get_dbi(avg, centres, dist_metric=dist_metric)

    """
    for i in range(k):
        plt.scatter(dataset[cluster[i], 0], dataset[cluster[i], 1], c=COLORS[i])
    plt.scatter(initial_centres[:, 0], initial_centres[:, 1], s=100, c='black', marker='+')
    plt.scatter(centres[:,0], centres[:,1], s=100, c='red', marker='x')
    plt.show()
    plt.close()
    """
    return centres, result, dbi


def ada_k_means(dataset, dist_metric=cos_dist):
    m, n = dataset.shape
    k = int(m ** 0.5)

    centres, result, dbi = k_means(dataset, k+1, dist_metric=dist_metric)

    min_dist = np.inf
    for i in range(len(centres)):
        for j in range(i, len(centres)):
            dist = dist_metric(centres[i], centres[j])
            if dist < min_dist:
                min_dist = dist
                merge_centre = (i, j)

    cur_centres = list(centres.copy())
    del cur_centres[merge_centre[1]]
    cur_centres = np.array(cur_centres)
    cur_centres, cur_result, cur_dbi = k_means(dataset, k, cur_centres, dist_metric=dist_metric)

    while True:
        min_dist = np.inf
        for i in range(len(cur_centres)):
            for j in range(i, len(cur_centres)):
                dist = dist_metric(cur_centres[i], cur_centres[j])
                if dist < min_dist:
                    min_dist = dist
                    merge_centre = (i, j)

        new_centres = list(cur_centres.copy())
        del new_centres[merge_centre[1]]
        new_centres = np.array(new_centres)
        new_centres, new_result, new_dbi = k_means(dataset, k-1, new_centres, dist_metric=dist_metric)

        if new_dbi > cur_dbi:
            cur_dbi = new_dbi
            cur_centres = new_centres
            cur_result = new_result
            k -= 1
        else:
            break
    print("k =", k)
    return cur_centres, cur_result


def k_folds_validation(dataset, labels, kf, dist_metric=cos_dist):
    index_list = list(range(len(dataset)))
    random.shuffle(index_list)
    eval_num = len(dataset)//kf
    for i in range(kf):
        head = i * eval_num
        tail = head + eval_num

        train_data = dataset[index_list[:head]+index_list[tail:]]
        eval_data = dataset[index_list[head:tail]]

        train_labels = labels[index_list[:head]+index_list[tail:]]
        eval_labels = labels[index_list[head:tail]]

        centres, train_result = ada_k_means(train_data, dist_metric=dist_metric)
        train_jc, train_fmi, train_ri = get_scores(train_labels, train_result)
        print("Jaccord Coefficient on the train data: {}\n"
              "FMI on the train data: {}\n"
              "Rand Index on the train data: {}\n".format(train_jc, train_fmi, train_ri))

        eval_result = test(eval_data, centres, dist_metric)
        eval_jc, eval_fmi, eval_ri = get_scores(eval_labels, eval_result)
        print("Jaccord Coefficient on the eval data: {}\n"
              "FMI on the eval data: {}\n"
              "Rand Index on the eval data: {}\n\n\n".format(eval_jc, eval_fmi, eval_ri))


def test(dataset, centres, dist_metric):
    m, n = dataset.shape
    result = np.array([0] * m)
    for i in range(m):
        min_dist = np.inf
        for j in range(len(centres)):
            dist = dist_metric(dataset[i], centres[j])
            if dist < min_dist:
                result[i] = j
                min_dist = dist
    """
    for i in range(len(centres)):
        plt.scatter(dataset[np.nonzero(result == i)[0], 0], dataset[np.nonzero(result == i)[0], 1], c=COLORS[i])
    plt.scatter(centres[:,0], centres[:,1], s=100, c='red', marker='x')
    plt.show()
    plt.close()
    """
    return result


def get_scores(labels, result):
    ss, sd, ds, dd = 0, 0, 0, 0
    m = len(labels)
    for i in range(m):
        for j in range(i+1, m):
            if result[i] == result[j]:
                if labels[i] == labels[j]:
                    ss += 1
                else:
                    sd += 1
            else:
                if labels[i] == labels[j]:
                    ds += 1
                else:
                    dd += 1
    jc = ss / (ss+sd+ds)
    fmi = ((ss/(ss+sd)) * (ss/(ss+ds)))**0.5
    ri = 2*(ss+dd) / (m*(m-1))
    return jc, fmi, ri


def load_watermelons():
    x = np.array([[0.697, 0.460],
                  [0.774, 0.376],
                  [0.634, 0.264],
                  [0.608, 0.318],
                  [0.556, 0.215],
                  [0.403, 0.237],
                  [0.481, 0.149],
                  [0.437, 0.211],
                  [0.666, 0.091],
                  [0.243, 0.267],
                  [0.245, 0.057],
                  [0.343, 0.099],
                  [0.639, 0.161],
                  [0.657, 0.198],
                  [0.360, 0.370],
                  [0.593, 0.042],
                  [0.719, 0.103],
                  [0.359, 0.188],
                  [0.339, 0.241],
                  [0.282, 0.257],
                  [0.748, 0.232],
                  [0.714, 0.346],
                  [0.483, 0.312],
                  [0.478, 0.437],
                  [0.525, 0.369],
                  [0.751, 0.489],
                  [0.532, 0.472],
                  [0.473, 0.376],
                  [0.725, 0.445],
                  [0.446, 0.459]])
    y = np.array([1]*8 + [0]*13 + [1]*9)
    return x, y


if __name__ == '__main__':
    x, y = load_watermelons()
    k_folds_validation(x, y, kf=3, dist_metric=cos_dist)
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import random


def euclid_dist(a, b):
    return np.sum((a[:-1] - b[:-1]) ** 2) ** 0.5


def euclid_with_vdm_dist(a, b, k, vdm_dict):
    euclid_part = np.sum((a[:-1] - b[:-1]) ** 2)
    vdm_part = 0
    for i in range(k):
        try:
            a_vdm = vdm_dict[i][a[-1]]
        except KeyError:
            a_vdm = 0
        try:
            b_vdm = vdm_dict[i][b[-1]]
        except KeyError:
            b_vdm = 0
        vdm_part += (a_vdm - b_vdm) ** 2
    return  (euclid_part + vdm_part) ** 0.5


def get_total_dist(data, cluster_data, k, vdm_dict):
    total_dist = 0
    for each in cluster_data:
        total_dist += euclid_with_vdm_dist(data, each, k, vdm_dict)
    return  total_dist


def normalization(input_x):
    max_value = np.max(input_x, 0)
    min_value = np.min(input_x, 0)
    range_value = max_value - min_value
    output_x = (input_x - min_value)/(range_value + 1e-7)
    return output_x


def get_vdm_dict(dataset, k, clusters):
    denom_dict = dict()
    for i in range(len(dataset)):
        value = dataset[i, -1]
        if value not in denom_dict:
            denom_dict[value] = 0
        denom_dict[value] += 1

    vdm_dict = {i: {} for i in range(k)}
    for i in range(k):
        for j in clusters[i]:
            value = dataset[j, -1]
            if value not in vdm_dict[i]:
                vdm_dict[i][value] = 0
            vdm_dict[i][value] += 1

    for i in range(k):
        for value in vdm_dict[i]:
            vdm_dict[i][value] /= denom_dict[value]

    return vdm_dict


def k_medoids(dataset, k, centres=None, max_iter=500):
    m, n = dataset.shape

    if centres is None:
        centres = random.sample(list(range(m)), k)

    clusters = [[] for i in range(k)]
    for i in range(m):
        min_dist = np.inf
        for j in range(k):
            dist = euclid_dist(dataset[i], dataset[centres[j]])
            if dist < min_dist:
                min_dist = dist
                best_cluster = j
        clusters[best_cluster].append(i)

    result = np.array([0] * m)

    for l in range(max_iter):
        print(centres)
        vdm_dict = get_vdm_dict(dataset, k, clusters)

        change = False
        for i in range(k):
            min_total_dist = np.inf
            for j in range(len(clusters[i])):
                total_dist = get_total_dist(dataset[clusters[i][j]], dataset[clusters[i]], k, vdm_dict)
                if total_dist < min_total_dist:
                    min_total_dist = total_dist
                    best_centre = clusters[i][j]
            if best_centre != centres[i]:
                centres[i] = best_centre
                change = True

        if change == False:
            break

        clusters = [[] for i in range(k)]

        for i in range(m):
            min_dist = np.inf
            for j in range(k):
                dist = euclid_with_vdm_dist(dataset[i], dataset[centres[j]], k, vdm_dict)
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = j
            clusters[best_cluster].append(i)
            result[i] = best_cluster

    return dataset[centres], result


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


def load_dataset():
    department = {'IT': 0,
                  'RandD': 1,
                  'accounting': 2,
                  'hr': 3,
                  'management': 4,
                  'marketing':5,
                  'product_mng': 6,
                  'sales': 7,
                  'support': 8,
                  'technical': 9}
    f = pd.read_csv('/data/hr-analytics.csv')
    m = len(f)
    dataset = np.zeros((m, 9))

    dataset[:, 0] = np.array(f['satisfaction_level'])
    dataset[:, 1] = np.array(f['last_evaluation'])
    dataset[:, 2] = np.array(f['number_projects'])
    dataset[:, 3] = np.array(f['average_monthly_hours'])
    dataset[:, 4] = np.array(f['time_spent_company'])
    dataset[:, 5] = np.array(f['work_accident'])
    dataset[:, 6] = np.array(f['promotion_last_5_years'])
    dataset[:, 7] = np.array(f['salary_level'])

    for i in range(m):
        dataset[i, 8] = department[f['department'][i]]

    labels = np.array(f['left'])

    return  dataset, labels


if __name__ == '__main__':
    dataset, labels = load_dataset()

    dataset[:, :-1] = normalization(dataset[:, :-1])

    centres, result = k_medoids(dataset, k=30)

    print(centres)
    print(result)

    jc, fmi, ri = get_scores(labels, result)

    print("Jaccord Coefficient on the train data: {}\n"
          "FMI on the train data: {}\n"
          "Rand Index on the train data: {}\n".format(jc, fmi, ri))
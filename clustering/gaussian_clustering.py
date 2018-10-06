# -*- coding: utf-8 -*-


import numpy as np
import random
from math import *


def get_gaussian_prob(x, mean, sigma, n):
    return exp(-0.5 * np.mat(x - mean) * sigma.I * np.mat(x - mean).T) / ((2 * pi)**(n/2) * np.linalg.det(sigma)**0.5)


def gaussian_clusters(x, k):
    m, n = x.shape

    mean_array = x[random.sample(list(range(m)), k)]
    alpha_array = np.ones(k) / k
    sigma_mats = [np.mat(np.eye(n)) / 10 for i in range(k)]
    gamma = np.zeros([m, k])

    for j in range(m):
        for i in range(k):
            gamma[j, i] = alpha_array[i] * get_gaussian_prob(x[j], mean_array[i], sigma_mats[i], n)
        gamma[j] /= np.sum(gamma[j])
    while True:
        cur_gamma = gamma.copy()

        for i in range(k):
            gamma_i = np.sum(gamma[:, i])

            mean_array[i] = np.sum(gamma[:, i].reshape([m, 1]) * x, 0) / gamma_i

            sigma_mats[i] = np.mat(np.zeros([n, n]))
            for j in range(m):
                sigma_mats[i] += gamma[j, i] * np.mat(x[j] - mean_array[i]).T * np.mat(x[j] - mean_array[i])
            sigma_mats[i] /= gamma_i

            alpha_array[i] = gamma_i / m

        for j in range(m):
            for i in range(k):
                gamma[j, i] = alpha_array[i] * get_gaussian_prob(x[j], mean_array[i], sigma_mats[i], n)
            gamma[j] /= np.sum(gamma[j])

        if np.sum((gamma - cur_gamma)**2) < 1e-5:
            break

    result = np.zeros(m)
    for j in range(m):
        result[j] = np.argmax(gamma[j])

    return alpha_array, mean_array, sigma_mats, result


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
    alpha_array, mean_array, sigma_mats, result = gaussian_clusters(x, k=3)
    print(result)
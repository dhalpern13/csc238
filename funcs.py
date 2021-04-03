from math import ceil

import numpy as np
from scipy.stats import bernoulli


def smallest_odd_larger(m):
    c = ceil(m)
    return c + (not c % 2)


def best_k_accuracies(expert_opinions, k):
    top_k = expert_opinions[-k:, ::]  # Numpy only natively sorts ascending, so we take the last k
    num_correct = top_k.sum(axis=0)
    majority_correct = num_correct > k / 2
    return np.sum(majority_correct)


def run_experiment(n, k_funcs, dist, batch_size):
    competencies = dist.rvs((n, batch_size))
    sorted_comps = np.sort(competencies, axis=0)
    expert_opinions = bernoulli(sorted_comps).rvs()
    return {k_name: best_k_accuracies(expert_opinions, k_func(n)) for k_name, k_func in k_funcs.items()}

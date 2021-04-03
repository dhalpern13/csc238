from math import ceil

import numpy as np
from scipy.stats import bernoulli, rv_continuous


def smallest_odd_larger(m: float) -> int:
    """Given a float m, return the smallest odd number larger than m."""
    c = ceil(m)
    return c + (not c % 2)


def best_k_accuracies(expert_opinions: np.ndarray, k: int) -> int:
    """
    Given an n x batch_size array of expert 0/1 judgements such that in each column,
    experts are sorted by ascending competence, compute the number of rows where a
    majority of the top k experts are correct.
    """
    # Restrict to k most accurate experts (numpy only natively sorts ascending, so we take the last k).
    top_k = expert_opinions[-k:, ::]
    # Compute how many correct experts in each instance.
    num_correct = top_k.sum(axis=0)
    # Whole instance is correct if at least k/2 out of top k correct.
    majority_correct = num_correct > k / 2
    # Add up total number of correct.
    return int(np.sum(majority_correct))


def run_experiment(n: int, k_vals: dict[str, int], dist: rv_continuous, batch_size: int) -> dict[str, int]:
    """
    Run an experiment with the given number of experts, k_vals, distribution, and batch size.

    :param n: Number of experts.
    :param k_vals: Dictionary with keys that are the names of the k values (i.e., "sqrt", "logn") and
                   values that are the actual k values.
    :param dist: Distribution to draw the expert competencies.
    :param batch_size: Number of instances to run at once (makes computation faster as one big array).
    :return: Dictionary with keys as k_value names, and values that are the number of instances
             (out of batch_size total),that the top k experts got the correct answer.
    """
    # Sample competencies
    competencies = dist.rvs((n, batch_size))
    # Sort by expert competencies
    sorted_comps = np.sort(competencies, axis=0)
    # Sample expert opinions from their competencies
    expert_opinions = bernoulli(sorted_comps).rvs()
    # Calculate number correct for each k.
    return {k_name: best_k_accuracies(expert_opinions, k_val) for k_name, k_val in k_vals.items()}

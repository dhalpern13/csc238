from collections import Counter
from datetime import datetime
from itertools import product
from math import log, sqrt

import pandas as pd
from scipy.stats import uniform, beta

from funcs import smallest_odd_larger, run_experiment

batch_size = 10000
iterations = 100

total_runs = batch_size * iterations

output_file = f'data/experiment-{total_runs}-runs-{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'

n_vals = [3, 10, 50, 200, 500]

k_vals = {
    'const1': lambda n: 1,
    'const3': lambda n: 3,
    'logn': lambda n: smallest_odd_larger(log(n)),
    'n^.4': lambda n: smallest_odd_larger(n ** .4),
    'sqrt': lambda n: smallest_odd_larger(sqrt(3 / 4 * n)),
    'n/2': lambda n: smallest_odd_larger(n / 2),
}

dists = {
    'uniform[0,1]': uniform(0, 1),
    'uniform[.1,.9]': uniform(.1, .8),  # scipy has weird syntax, first param is lower end and second is length
    'beta[2,2]': beta(2, 2),
}

if __name__ == '__main__':
    rows = []
    for (dist_name, dist), n in product(dists.items(), n_vals):
        counts = Counter()
        for i in range(iterations):
            if i % 10 == 0:
                print(f'{dist_name}, n={n}: Iteration {i}')
            counts += run_experiment(n, k_vals, dist, batch_size)
        row = {'dist': dist_name, 'n': n, **counts}
        rows.append(row)

    df = pd.DataFrame(rows, columns=['dist', 'n', *k_vals.keys()])
    df.to_csv(output_file, index=False)

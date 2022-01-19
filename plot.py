import json
from os import error
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
from util import get_lower_bound, get_alt_bound

MARKERS = {
    'PPR-Bernoulli-1v1': 'o',
    'A1-1v1': 's',
    'LUCB': '+',
    'KL-LUCB': '*',
    'KLSN-1v1': 'x',
    'A1-1vr': 'd',
    'KLSN-1vr': 'p',
    'PPR-Bernoulli-1vr' : '*'
}
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p1', type=float, default=0.65)
    parser.add_argument('--p2', type=float, default=0.35)
    args = parser.parse_args()
    p1 = args.p1
    p2 = args.p2

    sample_data = defaultdict(list)
    error_data = defaultdict(list)

    with open('data/data_p3.txt', 'r') as f:
        for line in f:
            line = json.loads(line)
            algorithm = line['algorithm']
            samples = line['num_samples']
            se = line['se']
            delta = line['delta']
            sample_data[algorithm].append((delta, samples))
            error_data[algorithm].append((delta, se))
    #Assumes all algos have been run for the same values of delta
    deltas = [delta for delta, samples in sorted(sample_data['A1-1v1'])]
    
    lower_bounds = []
    for delta in deltas:
        lower_bounds.append(get_lower_bound(p1, p2, delta))
    
    fig = plt.figure()
    ax = fig.add_subplot()
    for algorithm in sample_data.keys():
        sample_data[algorithm] = [samples for delta, samples in sorted(sample_data[algorithm])]
        error_data[algorithm] = [error for delta, error in sorted(error_data[algorithm])]
        ax.errorbar(deltas, sample_data[algorithm], yerr=error_data[algorithm], label=algorithm, marker=MARKERS[algorithm])
    ax.plot(deltas, lower_bounds, label='Lower Bound', marker='d')
    ax.set_xscale('log')
    plt.legend()
    plt.show()

    plot_ratio = True
    if not plot_ratio:
        return
    fig_ratio = plt.figure()
    ax = fig_ratio.add_subplot()
    for algorithm in sample_data.keys():
        ratios = [val / bound for val, bound in zip(sample_data[algorithm], lower_bounds)]
        ax.plot(deltas, ratios, label=algorithm, marker=MARKERS[algorithm])
        print(f'============{algorithm}=======================')
        for delta, ratio in zip(deltas, ratios):
            print(f'{delta}\t{ratio}')
    one = [1 for _ in deltas]
    ax.plot(deltas, one, label='1')
    ax.set_xscale('log')
    plt.legend()
    plt.show()
    print(f'Delta\tRatio')

if __name__ == '__main__':
    main()
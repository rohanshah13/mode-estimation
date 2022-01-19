import subprocess
import argparse
import time

# ALGORITHMS = ['PPR\ 1v1', 'KL-SN\ 1v1', 'A1\ 1v1', 'LUCB', 'KL-LUCB']
ALGORITHMS = [ 'KL-SN\ 1vr', 'A1\ 1v1', 'KL-SN\ 1v1', 'PPR\ 1v1', 'A1\ 1vr']
# ALGORITHMS = ['PPR\ 1vr']
COMMAND = 'python3 {}/main.py -pd {} -d {} -rs {} --outfile {} --iterations {}'
DELTAS = [10**(-i) for i in range(1,202,20)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--distribution', default='650_350')
    parser.add_argument('--outfile', default='data/data_p3.txt')
    parser.add_argument('--iterations', default=100, type=int)
    args = parser.parse_args()

    distribution = [int(val) for val in args.distribution.split('_')]
    assert(sum(distribution) == 1000)

    for algo in ALGORITHMS:
        algo_tic = time.perf_counter()
        for delta in DELTAS:
            tic = time.perf_counter()
            subprocess.run(COMMAND.format(algo, args.distribution, delta, args.seed, args.outfile, args.iterations), shell=True)
            toc = time.perf_counter()
            print(f'Time taken for algorithm {algo} on delta {delta} for {args.iterations} iterations = {toc-tic} seconds')
        algo_toc = time.perf_counter()
        print(f'Total time taken by algorithm {algo} for {args.iterations} iterations = {algo_toc - algo_tic} seconds')
if __name__ == '__main__':
    main()
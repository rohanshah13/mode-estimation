import subprocess
import argparse

# ALGORITHMS = ['PPR\ 1v1', 'A1\ 1v1', 'KL-LUCB', 'KL-SN\ 1v1', 'LUCB']
ALGORITHMS = ['PPR\ 1v1', 'A1\ 1v1', 'KL-SN\ 1v1']
COMMAND = 'python3 {}/main_seed.py -pd {} -d {} -rs {} --outfile {} --iterations {}'
# DELTAS = [10**(-i) for i in range(1,202,10)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--distribution', default='650_350')
    parser.add_argument('--outfile', default='data/data_seed.txt')
    parser.add_argument('--iterations', default=100)
    parser.add_argument('--delta', type=float, default=0.1)
    args = parser.parse_args()

    for algo in ALGORITHMS:
        subprocess.run(COMMAND.format(algo, args.distribution, args.delta, args.seed, args.outfile, args.iterations), shell=True)

if __name__ == '__main__':
    main()
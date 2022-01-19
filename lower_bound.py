import numpy as np
import argparse

def get_lower_bound(p1, p2, delta):
    delta_factor = np.log(1/(2.4*delta))
    instance_factor = p1/((p1-p2)**2)
    return instance_factor*delta_factor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p1', default=0.65, type=float)
    args = parser.parse_args()
    p1 = args.p1
    p2 = 1 - p1

    for i in range(1,21):
        delta = 10**(-1*i)
        lower_bound = get_lower_bound(p1, p2, delta)
        print(f'Mistake Probability = {delta}')
        print(f'Lower Bound = {lower_bound}')    

if __name__ == '__main__':
    main()
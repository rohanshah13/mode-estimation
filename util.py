import json
import numpy as np

def write_data(algorithm, num_samples, se, delta, failure_rate, seed, outfile):
    json_data = {}
    json_data['algorithm'] = algorithm
    json_data['num_samples'] = num_samples
    json_data['se'] = se
    json_data['delta'] = delta
    json_data['failure_rate'] = failure_rate
    json_data['seed'] = seed
    with open(outfile, 'a') as f:
        f.write(json.dumps(json_data) + '\n')

def write_data_seed(algorithm, num_samples, delta, num_success, outfile):
    json_data = {}
    json_data['algorithm'] = algorithm
    json_data['samples'] = num_samples
    num_success = [int(x) for x in num_success]
    json_data['success'] = num_success
    # json_data['se'] = se
    json_data['delta'] = delta
    # json_data['failure_rate'] = failure_rate
    with open(outfile, 'a') as f:
        f.write(json.dumps(json_data) + '\n')

def get_lower_bound(p1, p2, delta):
    print(p1, p2, delta)
    delta_factor = np.log(1/(2.4*delta))
    # instance_factor = p1/((p1-p2)**2)
    mid = (p1 + p2)/2
    instance_factor = p1*np.log(p1/mid) + p2*np.log(p2/mid)
    instance_factor = 1/instance_factor
    return instance_factor*delta_factor

def get_our_bound(p1, p2, delta):
    delta_factor = np.log(2.49/(((p1-0.5)**2)*delta))
    instance_factor = 20.775*p1/((p1-0.5)**2)
    return delta_factor*instance_factor

def get_alt_bound(p1, p2, delta):
    delta_factor = np.log(1/(2.4*delta))
    instance_factor = kl_divergence(p1, p2)
    instance_factor = 1/instance_factor
    return delta_factor*instance_factor

def kl_divergence(p1, q1):
    return p1*np.log(p1/q1) + (1 - p1)*np.log((1-p1)/(1-q1))
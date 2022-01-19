import argparse
import numpy as np
from math import e
from scipy.stats import beta
import logging
import sys
#KLSN Stuff
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

#set parameter
def get_kl_lower_bound(p,beta, time):
		lo = 0
		hi = p
		q = (lo+hi)/2
		lhs = KL_div(p,q)*time

		while abs(beta-lhs) > 1e-5:
			if abs(hi-lo) < 1e-4:
				break
			if lhs > beta:
				lo = q
			else:
				hi = q
			q = (lo+hi)/2
			lhs = KL_div(p,q)*time
		return q

#set parameter
def get_kl_upper_bound(p,beta, time):
		lo = p
		hi = 1
		q = (lo+hi)/2
		lhs = KL_div(p,q)*time

		while abs(beta-lhs) > 1e-5:
			if abs(hi-lo) < 1e-4:
				break
			if lhs > beta:
				hi = q
			else:
				lo = q
			q = (lo+hi)/2
			lhs = KL_div(p,q)*time
		return q

#complete
def eval_func(delta_val):
	return 2*(e**2) * delta_val*np.exp(-1*delta_val)

#complete
def solve_equation(delta):
	left = 0
	right = 700
	mid = int((left+right)/2)

	itera = 0
	while abs( eval_func(mid) - delta) > 1e-300:

		if(abs(right - left)< 1e-9):
			break
		if(delta<eval_func(mid)):
			left = mid
		else:
			right = mid
		mid = (left+right)/2
		itera += 1
	
	return mid


#complete
def exploration_rate(time,new_delta):

	rate = new_delta*(1+np.log(new_delta))*np.log(np.log(time))/((new_delta-1)*np.log(new_delta)) + new_delta

	return rate

#complete
def KL_div(a, b):
		if a == 0:
			if b == 1:
				return float("inf")
			else:
				return (1-a)*np.log((1-a)/(1-b))
		elif a == 1:
			if b == 0:
				return float("inf")
			else:
				return a*np.log(a/b)
		else:
			if b == 0 or b == 1:
				return float("inf")
			else:
				return a*np.log(a/b) + (1-a)*np.log((1-a)/(1-b))


#A1 Stuff

#complete
def betaValue(currIterations, communitySamples, numParties, deltaValue):

	cs = communitySamples
	t = currIterations
	k = numParties
	delta = deltaValue

	# summation (Zp - Zq)^2 = cs * (t - cs)
	
	empiricalVariance = 1.0 / (t * (t - 1)) * (cs * (t - cs))

	logTerm = np.log(4 * k *  t * t / delta)
	term1 = np.sqrt(2 * empiricalVariance * logTerm / t)
	term2 = 7 * logTerm / (3 * (t - 1))
	beta = term1 + term2

	return beta

#PPR Stuff

# set parameter
# function for finding the confidence sequence at a particular t in horizon
def binary_confidence_sequence(params_init, params_final, step_size=0.01, alpha=0.05, e=1e-12):
	'''
		params_init: parameters of the prior beta distribution (list)
		params_final: parameters of the posterior beta distribution (list)
		step_size: For searching the parameter space
		alpha: error probability
	'''

	# We implement PPR-1vr by dividing our parameter space into 1/step_size parts and maintaining confidence sequences
	# This can be done more accurately by ternary searching a point in the confidence sequence (say x), and then 
	# binary searching in [0, x] and [x, 1] for the bounds of the confidence sequence
	# However, the above is slower than this approach

	# possible p values
	p_vals = np.linspace(0, 1, num=int(1 / step_size) + 1)
	indices = np.arange(len(p_vals))

	# computation of prior
	log_prior_0 = beta.logpdf(p_vals, params_init[0], params_init[1])
	# log_prior_1 = beta.logpdf(p_vals, params_init[1], params_init[0])

	# computation of posterior
	log_posterior_0 = beta.logpdf(p_vals, params_final[0], params_final[1])
	# log_posterior_1 = beta.logpdf(p_vals, params_final[1], params_final[0])

	# martingale computation
	log_martingale_0 = log_prior_0 - log_posterior_0
	# log_martingale_1 = log_prior_1 - log_posterior_1

	# Confidence intervals
	ci_condition_0 = log_martingale_0 < np.log(1 / alpha)
	# ci_condition_1 = log_martingale_1 < np.log(1 / alpha)
	
	ci_indices_0 = np.copy(indices[ci_condition_0])
	# ci_indices_1 = np.copy(indices[ci_condition_1])
	return [p_vals[np.min(ci_indices_0)], p_vals[np.max(ci_indices_0)]], [0,0]
	
	# return [p_vals[np.min(ci_indices_0)], p_vals[np.max(ci_indices_0)]], [p_vals[np.min(ci_indices_1)], p_vals[np.max(ci_indices_1)]]

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--T',default=10000, type=int)
	parser.add_argument('--delta', default=0.1, type=float)
	args = parser.parse_args()
	T = args.T
	delta = args.delta

	output_file_handler = logging.FileHandler(f'out_{T}_{delta}.log')
	stdout_handler = logging.StreamHandler(sys.stdout)
	logger.addHandler(output_file_handler)
	logger.addHandler(stdout_handler)

	klsn_delta = solve_equation(delta)
	for t in range(2,T):
		logger.debug(t)
		klsn_beta = exploration_rate(t, klsn_delta)
		for k in range(t+1):
			empirical_mean = k/t
			#Compute A1 bounds
			a1_beta = betaValue(t, k, 2, delta)
			a1_ucb = empirical_mean + a1_beta
			a1_lcb = empirical_mean - a1_beta
			
			#Compute KLSN bounds
			klsn_lcb = get_kl_lower_bound(empirical_mean, klsn_beta, t)
			klsn_ucb = get_kl_upper_bound(empirical_mean, klsn_beta, t)

			#Compute PPR bounds
			ppr_step_size = 1e-4
			ppr_params_init = [1, 1]
			ppr_params_final = [k+1, t-k+1]
			ci_list = binary_confidence_sequence(ppr_params_init, ppr_params_final, ppr_step_size, delta)
			ppr_lcb = ci_list[0][0]
			ppr_ucb = ci_list[0][1]

			#Verify that PPR is the tightest
			if a1_lcb > ppr_lcb:
				logger.debug(f't = {t}, k = {k}')
				logger.debug('A1 has a tighter lower bound')
			if klsn_lcb > ppr_lcb:
				logger.debug(f't = {t}, k = {k}')
				logger.debug('KLSN has a tigher lower bound')
			if a1_ucb < ppr_ucb:
				logger.debug(f't = {t}, k = {k}')
				logger.debug('A1 has a tighter upper bound')
			if klsn_ucb < ppr_ucb:
				logger.debug(f't = {t}, k = {k}')
				logger.debug('KLSN has a tighter upper bound')


if __name__ == '__main__':
	main()
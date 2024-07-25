import sys
import warnings
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from tqdm import tqdm

from examples.water_wave.numerics import single_simulation as simulation
from models.my_system import compute_drift_velocity as find_crossings

DATA_PATH = '../../data/water_wave/st_optimization.csv'
TOL = 1e-5

def main():
	"""
	This program finds the critical Stokes number at which a particle completes
	no more than one orbit in a linear wave before sinking to the seabed, and
	saves the results to the `data/water_wave` directory.
	"""
	# create lists to store results
	npe_list1, npe_list2 = [], []	# number of period endpoints
	St_list1 = [0.01, 0.1, 1, 10]	# Stokes numbers used without history
	St_list2 = St_list1.copy()		# Stokes numbers used with history

	# run initial simulations
	print('Computing the number of period endpoints for provided Stokes nums:')
	for St in tqdm(St_list1):
		results1, results2 = run(St, hide=True)
		npe1 = compute_npe(results1)
		npe2 = compute_npe(results2)
		npe_list1.append(npe1)
		npe_list2.append(npe2)

	# find starting points St_a and St_b
	print('Finding the starting points for the bisection method...')
	St_a1, St_b1 = find_starting_points(St_list1, npe_list1)
	St_a2, St_b2 = find_starting_points(St_list2, npe_list2)

	# if starting points were found, perform bisection method
	if St_a1 is not None:
		print('Performing bisection method for sims without history effects...')
		St_list1, npe_list1 = bisection_method(St_a1, St_b1, St_list1,
											   npe_list1, False)
		print('Performing bisection method for sims without history effects...',
			  'done.\n')
		print('Performing bisection method for sims with history effects...')
		St_list2, npe_list2 = bisection_method(St_a2, St_b2, St_list2,
											   npe_list2, True)
		print('Performing bisection method for sims with history effects...',
			  'done.')
		print('Writing results...', end='')

		# store results in a DataFrame
		history_list = [False] * len(St_list1) + [True] * len(St_list2)
		results = pd.DataFrame({'St': St_list1 + St_list2,
								'num_endpoints': npe_list1 + npe_list2,
								'history': history_list})
		# write results to data file
		filename = 'st_optimization.csv'
		results.to_csv(DATA_PATH, index=False)
		print('done.')

def run(St, hide=False):
	"""
	Runs a numerical simulation with the provided Stokes number.
	
	Parameters
	----------
	St : float
		The Stokes number to use for the initialization of the particle.
	hide : boolean, default=False
		Whether to hide the `tqdm` progress bar.

	Returns
	-------
	list, list
		The numerical solutions without and with history effects.
	"""
	# initialize variables for the simulations
	position, beta = (0, 0), 0.9
	depth, amplitude, wavelength = 10, 0.02, 1
	num_periods, delta_t = 5, 5e-3

	return simulation(position, St, depth, amplitude, wavelength, beta,
					  num_periods, delta_t, DATA_PATH, mode='r',
					  hide_progress=hide, crop=['x', 'z', 'xdot','t'])

def compute_npe(results):
	"""
	Computes the number of period endpoints produced in a simulation.

	Parameters
	----------
	results : list
		List of numerical solutions.

	Returns
	-------
	int
		The number of period endpoints in the simulation.
	"""
	x, z, xdot, t = results
	_, z_crossings, _, _, _ = find_crossings(x, z, xdot, t)
	return (z_crossings.size)

def find_starting_points(St_list, npe_list):
	"""
	Identifies the initial points to use for the bisection method.

	Parameters
	----------
	St_list : list
		A list of the Stokes numbers.
	npe_list : list
		The number of period endpoints corresponding to each Stokes number.

	Returns
	-------
	St_a, St_b : float
		The starting points for the bisection method.
	"""
	St_a, St_b, i = 0, 0, 0
	if len(St_list) < 2: # return if less than two Stokes nums are provided
		print('Error: cannot begin bisection method, too few Stokes numbers',
			  ' provided.')
		return None, None
	elif npe_list[0] <= 1: # return if there are no sims with multiple endpoints
		print('Error: cannot begin bisection method, no simulations have more',
			  ' than one period endpoint.')
		return None, None
	else:
		# iterate through the npe list until it drops below 2 endpoints
		while i < len(npe_list) and npe_list[i] > 1: i += 1
		St_a = St_list[i - 1]	# more than one period endpoint
		St_b = St_list[i]		# one or zero period endpoints

		if St_a == 0 and St_b == 0: # print an error if no points were found
			print('Error: could not find starting points for bisection method.')
			print(f'Stokes numbers:{stokes_nums}',
				  f'\nNumber of period endpoints:{npe}')
			return None, None
		else:
			print(f'Starting points found: St_a = {St_a}, St_b = {St_b}.')
			return St_a, St_b

def bisection_method(St_a, St_b, St_list, npe_list, history):
	"""
	Find the critical Stokes number using the bisection method.

	Parameters
	----------
	St_a, St_b : float
		The starting points for the bisection method.
	St_list : list
		The list of Stokes numbers.
	npe_list : list
		The number of period endpoints corresponding to each Stokes number.
	history : boolean
		Whether to include history effects.

	Returns
	-------
	St_list, npe_list : list
		Lists of the Stokes numbers and the corresponding number of period
		endpoints.
	"""
	St_c = (St_a + St_b) / 2 # initialize midpoint between St_a and St_b

	while (St_c - St_a) / St_c >= TOL:
		St_list.append(St_c)

		# compute number of period endpoints for St_c and add to npe_list
		results1, results2 = run(St_c)
		results = results1 if history == False else results2
		num_period_endpoints = compute_npe(results)
		npe_list.append(num_period_endpoints)

		# update either St_a or St_b and recompute St_c
		if num_period_endpoints > 1:
			St_a = St_c
		else:
			St_b = St_c
		St_c = (St_a + St_b) / 2

	# once a critical St is found, add it to the St_list and compute the npe
	St_list.append(St_c)
	results1, results2 = run(St_c)
	results = results1 if history == False else results2
	num_period_endpoints = compute_npe(results)
	npe_list.append(num_period_endpoints)

	return St_list, npe_list

if __name__ == '__main__':
	main()

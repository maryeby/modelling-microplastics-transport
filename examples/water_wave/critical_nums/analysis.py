import warnings
import pandas as pd
import numpy as np
from tqdm.contrib.itertools import product

from utils.data_tools import update_results
from utils.colors import print_success, print_failure
from transport_framework import particle as prt
from models.my_system import compute_drift_velocity as find_crossings
from models import water_wave as fl
from models import my_system as ts

# wave conditions
DEPTH = 10
AMPLITUDE = 0.02
WAVELENGTH = 1.5

X_0 = 0
SCALE = 2 / 3
BETAS = np.round(np.delete(np.arange(0.75, 1.23, 0.01), 25), 3)
NUM_PERIODS = 5
DELTA_T = 5e-3
TOL = 1e-5
CRITICAL_NPE = 1
HIDE_PROGRESS = True
OUT_FILE = '../../data/water_wave/critical_nums.csv'

def main():
	"""
	Identify critical Stokes numbers and density ratios.

	A Stokes number and density ratio pairing which causes a particle trajectory
	to be critically damped reveals a critical Stokes number and critical
	density ratio. Critical Stokes numbers are found for a range of critical
	density ratios with and without history, and results are saved to the
	`data/water_wave` directory.
	"""
	warnings.filterwarnings('ignore')
	results = {'beta_c': [], 'St_c': [], 'history': [], 'z_crossing_f': []}
	for beta, history in product(BETAS, [False, True]):
		h_str = 'with history effects' if history else 'without history effects'

		# create lists to store results
		npe_list = []	# number of period endpoints
		St_list = [0.01, 0.1, 1, 10, 100]

		# run initial simulations
		print('Computing the number of period endpoints for each',
			  'Stokes number...')
		for St in St_list:
			sols = run(St, beta, history)
			npe = compute_npe(sols)
			npe_list.append(npe)

		# find starting points St_a and St_b
		print('Finding the starting points for the bisection method...')
		St_a, St_b = find_starting_points(St_list, npe_list)

		# if starting points were found, perform bisection method
		if St_a is not None:
			print(f'Performing bisection method for sims {h_str}...')
			St_c, z_crossing_f = bisection_method(St_a, St_b, beta, history)
			results = update_results(results, [], [beta, St_c, history,
												   z_crossing_f])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

def run(St, beta, history):
	"""
	Run a numerical simulation under the specified conditions.
	
	Parameters
	----------
	St : float
		The Stokes number to use for the initialization of the particle.
	beta : float
		The ratio between the particle and fluid densities.
	history : bool
		Whether to include history effects

	Returns
	-------
	list
		A list of `ndarray`s including the particle position, horizontal
		velocity, and time.
	"""
	density_ratio = SCALE * beta
	particle = prt.Particle(St)
	wave = fl.WaterWave(DEPTH, AMPLITUDE, WAVELENGTH)
	system = ts.MyTransportSystem(particle, wave, density_ratio)
	z_0 = 0 if beta < 1 else -3
	xdot_0, zdot_0 = wave.velocity(X_0, z_0, t=0)
	t = np.arange(0, wave.period * NUM_PERIODS, DELTA_T)
	y = [X_0, z_0, xdot_0, zdot_0]
	x, z, xdot, _, t = system.maxey_riley(t, y, history, HIDE_PROGRESS)[:5]
	return [x, z, xdot, t]

def compute_npe(results, return_last=False):
	"""
	Compute the number of period endpoints produced in a simulation.

	Parameters
	----------
	results : list
		A list of `ndarray`s containing the simulation results.
	return_last : bool, default=False
		Whether to return the vertical position of the last 2 period endpoints.

	Returns
	-------
	int
		The number of period endpoints in the simulation.
	float (optional)
		The vertical position of the final period endpoint.
	"""
	h = 2 * np.pi * DEPTH / WAVELENGTH # dimensionless depth
	x, z, xdot, t = results
	_, z_crossings, _, _, _ = find_crossings(x, z, xdot, t)
	if return_last and z_crossings.size > 0:
		return z_crossings.size, z_crossings[-2:].tolist()
	elif return_last and z_crossings.size == 0:
		return z_crossings.size, []
	else:
		return z_crossings.size

def find_starting_points(St_list, npe_list):
	"""
	Identify the initial points to use for the bisection method.

	Parameters
	----------
	St_list : list
		A list of `float` elements, the Stokes numbers.
	npe_list : list
		A list of `int` elements, the number of period endpoints for each *St*.

	Returns
	-------
	St_a, St_b : float
		The starting points for the bisection method.
	"""
	St_a, St_b, i = 0, 0, 0
	if len(St_list) < 2: # return if less than two Stokes nums are provided
		print_failure('Cannot begin bisection method, too few Stokes numbers'
					+ ' provided.')
		return None, None
	elif all(npe <= CRITICAL_NPE for npe in npe_list):
		print_failure('Cannot begin bisection method, no simulations have more'
					+ ' than one period endpoint.')
		return None, None
	elif all(npe > CRITICAL_NPE for npe in npe_list):
		print_failure('Cannot begin bisection method, all simulations have more'
					+ ' than one period endpoint.')
		return None, None
	else:
		# iterate through the npe list until it drops below 2 endpoints
		while i < len(npe_list) - 1 and npe_list[i] > 1: i += 1
		St_a = St_list[i - 1]	# more than one period endpoint
		St_b = St_list[i]		# one or zero period endpoints

		if St_a == 0 and St_b == 0: # print an error if no points were found
			print_failure('Could not find starting points for the bisection '
						+ 'method.')
			print(f'\t\t Stokes numbers:{stokes_nums}',
				  f'\n\t\t Number of period endpoints:{npe}')
			return None, None
		else:
			print_success('Starting points found: ' \
					   + f'St_a = {St_a}, St_b = {St_b}.')
			return St_a, St_b

def bisection_method(St_a, St_b, beta, history):
	"""
	Find the critical Stokes number using the bisection method.

	Parameters
	----------
	St_a, St_b : float
		The starting points for the bisection method.
	beta : float
		The ratio between the particle and fluid densities.
	history : bool
		Whether to include history effects.

	Returns
	-------
	St_c : float
		The critical Stokes number.
	z_final : float
		The vertical position of the final period endpoint.
	"""
	z_final = None
	St_c = (St_a + St_b) / 2 # initialize midpoint between St_a and St_b
	while (St_c - St_a) / St_c >= TOL:
		# compute number of period endpoints for St_c
		results = run(St_c, beta, history)
		npe, final_endpoints = compute_npe(results, return_last=True)

		# ensure final point isn't too close to the seabed
		if final_endpoints:
			if len(final_endpoints) == 2:
				z_f1, z_f2 = final_endpoints
				if DEPTH + z_f2 < TOL:
					z_final = z_f1
					npe -= 1
				else:
					z_final = z_f2
			else: # len(final_endpoints) == 1
				z_final = final_endpoints[0]

		# update either St_a or St_b and recompute St_c
		if npe > CRITICAL_NPE:
			St_a = St_c
		else:
			St_b = St_c
		St_c = (St_a + St_b) / 2
	return St_c, z_final

if __name__ == '__main__':
	main()

import warnings
import pandas as pd
import numpy as np
from tqdm.contrib.itertools import product

from utils.data_tools import update_results
from utils.colors import print_success, print_failure
from transport_framework import particle as prt
from models.my_system import compute_drift_velocity as find_crossings
from models import stokes_wave as fl
from models import my_system as ts
from examples.linear_wave.critical_nums.analysis import DEPTH, WAVELENGTH, X_0,\
	 RS, ST_HATS, NUM_PERIODS, DELTA_T, TOL, CRITICAL_NPE

AMPLITUDE = 0.07
KEYS = ['R_c', 'St_c', 'history', 'z_crossing_f', 'A\'']
HIDE_PROGRESS = True
OUT_FILE = '../../data/stokes_wave/critical_nums.csv'

def main():
	"""
	Identify critical Stokes numbers and density ratios.

	A Stokes number and density ratio pairing which causes a particle trajectory
	to be critically damped reveals a critical Stokes number and critical
	density ratio. Critical Stokes numbers are found for a range of critical
	density ratios with and without history, and results are saved to the
	`data/stokes_wave` directory.
	"""
	warnings.filterwarnings('ignore')
	results = {key: [] for key in KEYS}
	for r, history in product(RS, [False, True]):
		h_str = 'with history effects' if history else 'without history effects'
		npe_list = []	# number of period endpoints

		# run initial simulations
		print('Computing the number of period endpoints for each',
			  'Stokes number...')
		for sthat in ST_HATS:
			sols = run(sthat, r, history)
			npe = compute_npe(sols)
			npe_list.append(npe)

		# find starting points St_a and St_b
		print('Finding the starting points for the bisection method...')
		sthat_a, sthat_b = find_starting_points(npe_list)

		# if starting points were found, perform bisection method
		if sthat_a is not None:
			print(f'Performing bisection method for sims {h_str}...')
			st_c, z_crossing_f = bisection_method(sthat_a, sthat_b, r, history)
			results = update_results(results, [], [r, st_c, history,
												   z_crossing_f, AMPLITUDE])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False, header=False, mode='a')

def run(sthat, r, history):
	"""
	Run a numerical simulation under the specified conditions.
	
	Parameters
	----------
	sthat : float
		The Stokes number to use for the initialization of the particle.
	r : float
		The ratio between the particle and fluid densities.
	history : bool
		Whether to include history effects

	Returns
	-------
	list
		A list of `ndarray`s including the particle position, horizontal
		velocity, and time.
	"""
	particle = prt.Particle(sthat)
	wave = fl.StokesWave(DEPTH, AMPLITUDE, WAVELENGTH)
	system = ts.MyTransportSystem(particle, wave, r)
	z_0 = 0 if r < 2 / 3 else -2
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

def find_starting_points(npe_list):
	r"""
	Identify the initial points to use for the bisection method.

	Parameters
	----------
	npe_list : list
		A list of `int` elements, the number of period endpoints for each $St$.

	Returns
	-------
	sthat_a, sthat_b : float
		The starting points for the bisection method.
	"""
	sthat_a, sthat_b, i = 0, 0, 0
	if len(ST_HATS) < 2: # return if less than two Stokes nums are provided
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
		sthat_a = ST_HATS[i - 1]	# more than one period endpoint
		sthat_b = ST_HATS[i]		# one or zero period endpoints

		if sthat_a == 0 and sthat_b == 0: # print error if no points were found
			print_failure('Could not find starting points for the bisection '
						+ 'method.')
			print(f'\t\t Stokes numbers:{ST_HATS}',
				  f'\n\t\t Number of period endpoints:{npe}')
			return None, None
		else:
			print_success('Starting points found: ' \
					   + f'Sthat_a = {sthat_a}, Sthat_b = {sthat_b}.')
			return sthat_a, sthat_b

def bisection_method(sthat_a, sthat_b, r, history):
	r"""
	Find the critical Stokes number using the bisection method.

	Parameters
	----------
	sthat_a, sthat_b : float
		The starting points for the bisection method.
	r : float
		The ratio between the particle and fluid densities.
	history : bool
		Whether to include history effects.

	Returns
	-------
	st_c : float
		The critical Stokes number $St_c$.
	z_final : float
		The vertical position of the final period endpoint.
	"""
	z_final = None
	sthat_c = (sthat_a + sthat_b) / 2 # initialize the midpoint
	while (sthat_c - sthat_a) / sthat_c >= TOL:
		# compute number of period endpoints for sthat_c
		results = run(sthat_c, r, history)
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

		# update either sthat_a or sthat_b and recompute sthat_c
		if npe > CRITICAL_NPE:
			sthat_a = sthat_c
		else:
			sthat_b = sthat_c
		sthat_c = (sthat_a + sthat_b) / 2

	# compute the critical Stokes number from the critical Stokes hat
	particle = prt.Particle(sthat_c)
	wave = fl.StokesWave(DEPTH, AMPLITUDE, WAVELENGTH)
	system = ts.MyTransportSystem(particle, wave, r)
	st_c = system.stokes_num
	return st_c, z_final

if __name__ == '__main__':
	main()

import sys 
import warnings
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from parallelbar import progress_starmap

from transport_framework import particle as prt
from models import water_wave as fl
from models import my_system as ts

DATA_PATH = '../data/water_wave/'

def main():
	"""
	This program produces numerical solutions for the motion of inertial
	particles in linear water waves and saves the results to the
	`data/water_wave` directory.
	"""
	# initialize parameters
	stokes_num = 0.01
	beta = 0.8
	depth, amplitude, wavelength = 10, 0.02, 1
	num_periods, delta_t = 50, 5e-3
	my_wave = fl.WaterWave(depth, amplitude, wavelength)

	# run the simulations with the appropriate function
	if type(stokes_num) is list:
		multi_stokes_nums(stokes_num, my_wave, beta, num_periods, delta_t)	
	elif type(beta) is list:
		multi_betas(stokes_num, my_wave, beta, num_periods, delta_t)	
	else:
		single_simulation(stokes_num, my_wave, beta, num_periods, delta_t)

def single_simulation(stokes_num, wave, beta, num_periods, delta_t):
	"""
	Runs a numerical simulation without history effects and a numerical
	simulation with history effects.

	Parameters
	----------
	stokes_num : float
		The Stokes number to use for the initialization of the particle.
	wave : Wave (obj)
		The wave through which the particle is transported.
	beta : float
		The ratio between the particle and fluid densities.
	num_periods : int
		The number of wave periods to integrate over.
	delta_t : float
		The size of the time steps used for integration.
	"""
	# ask the user whether to write the results or plot results without writing
	answer = input('(W)rite the results to the data file or (P)lot the results'
				   ' without writing? ')
	write_results = True if answer.upper() == 'W' else False

	# run simulation without history
	results = run_numerics(stokes_num, wave, beta, False, num_periods, delta_t,
						   hide_progress=False)

	# only write results if they contain no NaNs (skip failed simulations)
	if write_results and results['x'] is not None:
		write_data(results)
		history_results = run_numerics(stokes_num, wave, beta, True,
									   num_periods, delta_t)
		if history_results['x'] is not None:
			write_data(history_results)

	# only plot results if they contain no NaNs (skip failed simulations)
	elif not write_results and results['x'] is not None:
		# initialize plot
		plt.figure()
		plt.title(r'Particle Trajectory', fontsize=18)
		plt.xlabel('x', fontsize=14)
		plt.ylabel('z', fontsize=14)
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.minorticks_on()

		# plot results without history
		plt.plot('x', 'z', c='k', data=results, label='without history')
		history_results = run_numerics(stokes_num, wave, beta, True,
									   num_periods, delta_t)
		# plot results with history
		if history_results['x'] is not None:
			print('Plotting...')
			plt.plot('x', 'z', c='k', ls='--', data=history_results,
					 label='with history')
			plt.legend()
			plt.show()

def multi_stokes_nums(stokes_nums, wave, beta, num_periods, delta_t):
	"""
	Runs a numerical simulation without history effects and a numerical
	simulation with history effects for each Stokes number, and writes the
	results to a data file.

	Parameters
	----------
	stokes_nums : list
		A list of the Stokes numbers to use for the particle initializations.
	wave : Wave (obj)
		The wave through which the particle is transported.
	betas : float
		The ratio between the particle and fluid densities.
	num_periods : int
		The number of wave periods to integrate over.
	delta_t : float
		The size of the time step used for integration.
	"""
	num_tasks = len(stokes_nums)

	# run simulations without history effects
	params = zip(stokes_nums, itertools.repeat(wave), itertools.repeat(beta),
				 itertools.repeat(False), itertools.repeat(num_periods),
				 itertools.repeat(delta_t))
	results = progress_starmap(run_numerics, params, total=num_tasks)

	# remove results that contain NaNs (failed simulations)
	for i in range(len(results)):
		if results[i]['x'] is None:
			stokes_nums[i] = None
			results[i] = None
			num_tasks -= 1
	results = [i for i in results if i is not None]
	stokes_nums = [i for i in stokes_nums if i is not None]

	# run simulations with history effects unless non-history simulations failed
	history_results = []
	if num_tasks != 0:
		for i in range(num_tasks):
			print(f'\nRunning simulation {i + 1}/{num_tasks}...')
			result = run_numerics(stokes_nums[i], wave, beta, True, num_periods,
								  delta_t)
			history_results.append(result)
	
	# remove results that contain NaNs (failed simulations)
	for i in range(len(history_results)):
		if history_results[i]['x'] is None:
			results[i] = None
			history_results[i] = None
	results += history_results
	results = [i for i in results if i is not None]
	
	# write successful results to data file, if there are any
	if results:
		for result in results:
			write_data(result)

def multi_betas(stokes_num, wave, betas, num_periods, delta_t):
	"""
	Runs a numerical simulation without history effects and a numerical
	simulation with history effects for each value of beta.

	Parameters
	----------
	stokes_num : float
		The Stokes number to use for the initialization of the particle.
	wave : Wave (obj)
		The wave through which the particle is transported.
	betas : list
		A list of the ratios between the particle and fluid densities.
	num_periods : int
		The number of wave periods to integrate over.
	delta_t : float
		The size of the time step used for integration.
	"""
	num_tasks = len(betas)

	# run simulations without history effects
	params = zip(itertools.repeat(stokes_num), itertools.repeat(wave),
				 betas, itertools.repeat(False),
				 itertools.repeat(num_periods), itertools.repeat(delta_t))
	results = progress_starmap(run_numerics, params, total=num_tasks)

	# remove results that contain NaNs (failed simulations)
	for i in range(len(results)):
		if results[i]['x'] is None:
			betas[i] = None
			results[i] = None
			num_tasks -= 1
	results = [i for i in results if i is not None]
	betas = [i for i in betas if i is not None]

	# run simulations with history effects unless non-history simulations failed
	history_results = []
	if num_tasks != 0:
		for i in range(num_tasks):
			print(f'\nRunning simulation {i + 1}/{num_tasks}...')
			result = run_numerics(stokes_num, wave, betas[i], True, num_periods,
								  delta_t)
			history_results.append(result)
	
	# remove results that contain NaNs (failed simulations)
	for i in range(len(history_results)):
		if history_results[i]['x'] is None:
			results[i] = None
			history_results[i] = None
	results += history_results
	results = [i for i in results if i is not None]
	
	# write successful results to data file, if there are any
	if results:
		for result in results:
			write_data(result)

def run_numerics(stokes_num, wave, beta, include_history, num_periods, delta_t,
				 hide_progress=True):
	"""
	Runs a numerical simulation for the specified parameters.

	Parameters
	----------
	stokes_num : float
		The Stokes number to use for the initialization of the particle.
	wave : Wave (obj)
		The wave through which the particle is transported.
	density_ratio : float
		The ratio between the particle and fluid densities.
	include_history : boolean
		Whether to include history effects.
	num_periods : int
		The number of wave periods to integrate over.
	delta_t : float
		The size of the time step used for integration.
	hide_progress : bool, default=True
		Whether to hide the `tqdm` progress bar.

	Returns
	-------
	dict
		Dictionary containing the numerical solutions.
	"""
	# initialize parameters
	x_0, z_0 = 0, 0
	xdot_0, zdot_0 = wave.velocity(x_0, z_0, t=0)
	R = 2 / 3 * beta

	# initialize the particle and transport system
	my_particle = prt.Particle(stokes_num)
	my_system = ts.MyTransportSystem(my_particle, wave, R)

	# run numerics
	x, z, xdot, zdot, t = my_system.run_numerics(include_history,
												 x_0, z_0, xdot_0, zdot_0,
												 num_periods, delta_t,
												 hide_progress)
	# get useful wave conditions and the particle Reynolds number
	h = wave.depth
	A = wave.amplitude
	wavelength = wave.wavelength
	k = wave.wavenum
	omega = wave.angular_freq
	Fr = wave.froude_num
	U = wave.max_velocity
	general_Re_p = my_system.reynolds_num
	Re_p = None

	# organize results in a dictionary
	results = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t, 'z_0': z_0,
			   'St': stokes_num, 'beta': beta, 'history': include_history,
			   'h\'': h, 'A\'': A, 'wavelength\'': wavelength, 'k\'': k,
			   'omega\'': omega, 'Fr': Fr, 'U\'': U,
			   'general_Re_p': general_Re_p, 'Re_p': Re_p,
			   'num_periods\'': num_periods, 'delta_t\'': delta_t}

	# check if solutions contain any infinite values
	if np.isinf([x, z, xdot, zdot]).any():
		# print the failed simulation and the values of its parameters
		beta_char, lambda_char, pi_char, omega_char, deltat_char = '\u03B2', \
									'\u03BB', '\u03C0', '\u03C9', '\u0394'+'t'
		prime = '\''
		print('\nSimulation failed')
		print(f'{"z_0":10}{z_0:^10.4g}')
		print(f'{"St":10}{stokes_num:^10.4g}')
		print(f'{beta_char:10}{beta:^10.4g}')
		print(f'{"history":10}{include_history!s:^10}')
		print(f'{"h", prime:10}{h:^10}')
		print(f'{"A", prime:10}{A:^10.4g}')
		print(f'{lambda_char, prime:10}{wavelength:^10}')

		# print k as a multiple of pi if possible
		if k % np.pi == 0:
			k = f'{int(k // np.pi):d}{pi_char}'
			print(f'{"k", prime:10}{k:^10}')
		else:
			print(f'{"k", prime:10}{k:^10.4f}')

		print(f'{omega_char, prime:10}{omega.item():^10.4f}')
		print(f'{"Fr":10}{Fr.item():^10.4f}')
		print(f'{"U", prime:10}{U.item():^10.4f}')
		print(f'{"periods", prime:10}{num_periods:^10}')
		print(f'{deltat_char, prime:10}{delta_t:^10.4g}')
		print(f'{"Re_p":10}{general_Re_p:^10.4g}\n')
		# set the results of the failed simulation to None
		results = {key: None for key, value in results.items()}

	else:
		# compute the Reynolds number for each timestep
		u = wave.velocity(x, z, t) * U
		v = np.array([xdot, zdot]) * U
		Re_p = (2 * np.linalg.norm(v - u, axis=0) * np.sqrt(9 * stokes_num 
				  / (2 * k ** 2 * wave.reynolds_num))) \
				  / wave.kinematic_viscosity

		# organize data
		for key, value in list(results.items())[5:]:
			if key == 'Re_p':
				results['Re_p'] = Re_p
			else:
				value = [value] * len(x)
	return results

def write_data(results):
	"""Writes the provided `results` dictionary to a csv file."""
	filename = 'numerics.csv'
	df = pd.DataFrame(results)
	numerics = df.explode(list(df.columns.values), ignore_index=True)
	numerics.to_csv(DATA_PATH + filename, mode='a', header=False, index=False)
	print(f'Data added to {filename}.')

if __name__ == '__main__':
	main()

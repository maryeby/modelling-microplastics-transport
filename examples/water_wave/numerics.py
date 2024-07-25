import sys 
import os
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

def single_simulation(position, stokes_num, depth, amplitude, wavelength, beta,
					  num_periods, delta_t, filepath, mode='w', crop=[],
					  hide_progress=False):
	"""
	Runs a numerical simulation without history effects and a numerical
	simulation with history effects.

	Parameters
	----------
	position : tuple
		The initial (x_0, z_0) position of the particle.
	stokes_num : float
		The Stokes number to use for the initialization of the particle.
	depth : float
		The depth of the fluid *h'*.
	amplitude : float
		The amplitude of the wave *A'*.
	wavelength : float
		The wavelength *λ'*.
	beta : float
		The ratio between the particle and fluid densities.
	num_periods : int
		The number of wave periods to integrate over.
	delta_t : float
		The size of the time steps used for integration.
	filepath : str
		The path of the file where results will be stored.
	mode : str, default='w'
		How to handle the results: either write 'w', plot 'p', or return 'r'.
	crop : list, default=[]
		List of the variables to include in the results (includes all if empty).
	hide_progress : boolean, default=False
		Whether to hide the `tqdm` progress bar.
	"""
	# run simulation without history
	wave = fl.WaterWave(depth, amplitude, wavelength)
	results1 = run_simulation(position, stokes_num, wave, beta, False,
							  num_periods, delta_t, hide_progress)
	if results1['x'] is None:
		print('Simulation failed without history effects.')
	else:
		if mode == 'w': write_data(results1, filepath)
		results2 = run_simulation(position, stokes_num, wave, beta, True,
								  num_periods, delta_t, hide_progress)
		if results2['x'] is None:
			print('Simulation failed with history effects.')
		else: # both simulations succeeded
			if mode == 'w':
				write_data(results2, filepath)
			elif mode == 'p':
				plot_trajectory(results1, results2)
				path_list = filepath.split('/')
				filename = path_list[-1]
				user_input = input(f'Write data to {filename}? (Y/N): ')
				if user_input.upper() == 'Y':
					write_data(results1, filepath)
					write_data(results2, filepath)
			elif mode == 'r':
				if not crop:
					return list(results1.values()), list(results2.values())
				else:
					return [results1[key] for key in crop], \
						   [results2[key] for key in crop]
			else:
				print('Error: mode incorrectly specified.')
				print('Mode options: write \'w\', plot \'p\', or return \'r\'.')

def multi_stokes_nums(position, stokes_nums, depth, amplitude, wavelength, beta,
					  num_periods, delta_t, filepath):
	"""
	Runs a numerical simulation without history effects and a numerical
	simulation with history effects for each Stokes number, and writes the
	results to a data file.

	Parameters
	----------
	position : tuple
		The initial (x_0, z_0) position of the particle.
	stokes_nums : list
		A list of the Stokes numbers to use for the particle initializations.
	depth : float
		The depth of the fluid *h'*.
	amplitude : float
		The amplitude of the wave *A'*.
	wavelength : float
		The wavelength *λ'*.
	beta : float
		The ratio between the particle and fluid densities.
	num_periods : int
		The number of wave periods to integrate over.
	delta_t : float
		The size of the time step used for integration.
	filepath : str
		The path of the file where results will be stored.
	"""
	tasks = len(stokes_nums)
	wave = fl.WaterWave(depth, amplitude, wavelength)

	# run simulations without history effects
	params = zip(itertools.repeat(position), stokes_nums,
				 itertools.repeat(wave), itertools.repeat(beta),
				 itertools.repeat(False), itertools.repeat(num_periods),
				 itertools.repeat(delta_t))
	results1 = progress_starmap(run_simulation, params, n_cpu=4, total=tasks)

	# remove results that contain NaNs (failed simulations)
	for i in range(len(results1)):
		if results1[i]['x'] is None:
			stokes_nums[i] = None
			results1[i] = None
			tasks -= 1
	results1 = [i for i in results1 if i is not None]
	stokes_nums = [i for i in stokes_nums if i is not None]

	# run simulations with history effects unless non-history simulations failed
	results2 = []
	if tasks != 0:
		for i in range(tasks):
			print(f'\nRunning simulation {i + 1}/{tasks}...')
			result = run_simulation(position, stokes_nums[i], wave, beta, True,
								    num_periods, delta_t)
			results2.append(result)
	
	# remove results that contain NaNs (failed simulations)
	for i in range(len(results2)):
		if results2[i]['x'] is None or results2[i]['x'].size == 1:
			print_failed_simulation(results2[i])
			results1[i] = None
			results2[i] = None
	results1 += results2
	results1 = [i for i in results1 if i is not None]
	
	# write successful results to data file, if there are any
	if results1:
		for result in results1:
			write_data(result, filepath)

def multi_betas(position, stokes_num, depth, amplitude, wavelength, betas,
				num_periods, delta_t, filepath):
	"""
	Runs a numerical simulation without history effects and a numerical
	simulation with history effects for each value of beta.

	Parameters
	----------
	position : tuple
		The initial (x_0, z_0) position of the particle.
	stokes_num : float
		The Stokes number to use for the initialization of the particle.
	depth : float
		The depth of the fluid *h'*.
	amplitude : float
		The amplitude of the wave *A'*.
	wavelength : float
		The wavelength *λ'*.
	betas : list
		A list of the ratios between the particle and fluid densities.
	num_periods : int
		The number of wave periods to integrate over.
	delta_t : float
		The size of the time step used for integration.
	filepath : str
		The path of the file where results will be stored.
	"""
	tasks = len(betas)
	wave = fl.WaterWave(depth, amplitude, wavelength)

	# run simulations without history effects
	params = zip(itertools.repeat(position), itertools.repeat(stokes_num),
				 itertools.repeat(wave),
				 betas, itertools.repeat(False),
				 itertools.repeat(num_periods), itertools.repeat(delta_t))
	results1 = progress_starmap(run_simulation, params, n_cpu=4, total=tasks)

	# remove results that contain NaNs (failed simulations)
	for i in range(len(results1)):
		if results1[i]['x'] is None:
			betas[i] = None
			results1[i] = None
			tasks -= 1
	results1 = [i for i in results1 if i is not None]
	betas = [i for i in betas if i is not None]

	# run simulations with history effects unless non-history simulations failed
	results2 = []
	if tasks != 0:
		for i in range(tasks):
			print(f'\nRunning simulation {i + 1}/{tasks}...')
			result = run_simulation(position, stokes_num, wave, betas[i], True,
								    num_periods, delta_t)
			results2.append(result)
	
	# remove results that contain NaNs (failed simulations)
	for i in range(len(results2)):
		if results2[i]['x'] is None or results2[i]['x'].size == 1:
			print_failed_simulation(results2[i])
			results1[i] = None
			results2[i] = None
	results1 += results2
	results1 = [i for i in results1 if i is not None]
	
	# write successful results to data file, if there are any
	if results1:
		for result in results1:
			write_data(result, filepath)

def multi_positions(x_0s, stokes_num, depth, amplitude, wavelength, beta,
					num_periods, delta_t, filepath):
	"""
	Runs a numerical simulation without history effects and a numerical
	simulation with history effects for each value of beta.

	Parameters
	----------
	x_0s : list of tuples
		The initial (x_0, z_0) positions of the particle.
	stokes_num : float
		The Stokes number to use for the initialization of the particle.
	depth : float
		The depth of the fluid *h'*.
	amplitude : float
		The amplitude of the wave *A'*.
	wavelength : float
		The wavelength *λ'*.
	beta : float
		The ratio between the particle and fluid densities.
	num_periods : int
		The number of wave periods to integrate over.
	delta_t : float
		The size of the time step used for integration.
	filepath : str
		The path of the file where results will be stored.
	"""
	tasks = len(x_0s)
	wave = fl.WaterWave(depth, amplitude, wavelength)

	# run simulations without history effects
	params = zip(x_0s, itertools.repeat(stokes_num), itertools.repeat(wave),
				 itertools.repeat(beta), itertools.repeat(False),
				 itertools.repeat(num_periods), itertools.repeat(delta_t))
	results1 = progress_starmap(run_simulation, params, n_cpu=4, total=tasks)

	# remove results that contain NaNs (failed simulations)
	for i in range(len(results1)):
		if results1[i]['x'] is None:
			x_0s[i] = None
			results1[i] = None
			tasks -= 1
	results1 = [i for i in results1 if i is not None]
	x_0s = [i for i in x_0s if i is not None]

	# run simulations with history effects unless non-history simulations failed
	results2 = []
	if tasks != 0:
		for i in range(tasks):
			print(f'\nRunning simulation {i + 1}/{tasks}...')
			result = run_simulation(x_0s[i], stokes_num, wave, beta, True,
								    num_periods, delta_t)
			results2.append(result)
	
	# remove results that contain NaNs (failed simulations)
	for i in range(len(results2)):
		if results2[i]['x'] is None or results2[i]['x'].size == 1:
			print_failed_simulation(results2[i])
			results1[i] = None
			results2[i] = None
	results1 += results2
	results1 = [i for i in results1 if i is not None]
	
	# write successful results to data file, if there are any
	if results1:
		for result in results1:
			write_data(result, filepath)

def run_simulation(position, stokes_num, wave, beta, include_history,
				   num_periods, delta_t, hide_progress=True):
	"""
	Runs a numerical simulation for the specified parameters.

	Parameters
	----------
	position : tuple
		The initial (x_0, z_0) position of the particle.
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
	hide_progress : boolean, default=True
		Whether to hide the `tqdm` progress bar.

	Returns
	-------
	dict
		Dictionary containing the numerical solutions.
	"""
	# initialize parameters
	x_0, z_0 = position
	xdot_0, zdot_0 = wave.velocity(x_0, z_0, t=0)
	R = 2 / 3 * beta

	# initialize the particle and transport system
	my_particle = prt.Particle(stokes_num)
	my_system = ts.MyTransportSystem(my_particle, wave, R)

	# run numerics
	x, z, xdot, zdot, t, fpg_x, fpg_z, buoyancy_x, buoyancy_z, \
	   added_mass_x, added_mass_z, stokes_drag_x, stokes_drag_z, \
	   history_force_x, \
	   history_force_z = my_system.run_numerics(include_history, x_0, z_0,
												xdot_0, zdot_0, num_periods,
												delta_t, hide_progress,
												include_forces=True)
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
	results = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t,
			   'fluid_pressure_gradient_x': fpg_x,
			   'fluid_pressure_gradient_z': fpg_z,
			   'buoyancy_force_x': buoyancy_x, 'buoyancy_force_z': buoyancy_z,
			   'added_mass_force_x': added_mass_x,
			   'added_mass_force_z': added_mass_z,
			   'stokes_drag_x': stokes_drag_x, 'stokes_drag_z': stokes_drag_z,
			   'history_force_x': history_force_x, \
			   'history_force_z': history_force_z,
			   'x_0': x_0, 'z_0': z_0, 'St': stokes_num, 'beta': beta,
			   'history': include_history, 'h\'': h, 'A\'': A,
			   'wavelength\'': wavelength, 'k\'': k, 'omega\'': omega, 'Fr': Fr,
			   'U\'': U, 'general_Re_p': general_Re_p, 'Re_p': Re_p,
			   'num_periods\'': num_periods, 'delta_t\'': delta_t}

	# check if solutions contain any infinite values
	if np.isinf([x, z, xdot, zdot]).any():
		print_failed_simulation(results)
		results = {key: None for key, value in results.items()}
	else:
		# compute the Reynolds number for each timestep
		u = wave.velocity(x, z, t) * U
		v = np.array([xdot, zdot]) * U
		Re_p = (2 * np.linalg.norm(v - u, axis=0) * np.sqrt(9 * stokes_num 
				  / (2 * k ** 2 * wave.reynolds_num))) \
				  / wave.kinematic_viscosity

		# organize data
		for key, value in list(results.items())[15:]:
			if key == 'Re_p':
				results['Re_p'] = Re_p
			else:
				results[key] = np.repeat(value, x.size)
	return results

def print_failed_simulation(results):
	"""Prints the failed simulation and the values of its parameters."""
	beta_char, lambda_char, pi_char = '\u03B2', '\u03BB', '\u03C0'
	omega_char, delta_t_char = '\u03C9', '\u0394'+'t'
	get = lambda name : results[name]

	# extract parameter values from results
	x_0, z_0 = get('x_0'), get('z_0')
	stokes_num, beta = get('St'), get('beta')
	include_history = get('history')
	h, A, wavelength = get('h\''), get('A\''), get('wavelength\'')
	k, omega = get('k\''), get('omega\'')
	Fr, U = get('Fr'), get('U\'')
	num_periods, delta_t = get('num_periods\''), get('delta_t\'')
	general_Re_p = get('general_Re_p')

	prime = '\''
	print('\nSimulation failed')
	print(f'{"x_0":10}{x_0:^10.4g}')
	print(f'{"z_0":10}{z_0:^10.4g}')
	print(f'{"St":10}{stokes_num:^10.4g}')
	print(f'{beta_char:10}{beta:^10.4g}')
	print(f'{"history":10}{include_history!s:^10}')
	print(f'{"h" + prime:10}{h:^10}')
	print(f'{"A" + prime:10}{A:^10.4g}')
	print(f'{lambda_char + prime:10}{wavelength:^10}')

	# print k as a multiple of pi if possible
	if k % np.pi == 0:
		k = f'{int(k // np.pi):d}{pi_char}'
		print(f'{"k" + prime:10}{k:^10}')
	else:
		print(f'{"k" + prime:10}{k:^10.4f}')

	print(f'{omega_char + prime:10}{omega.item():^10.4f}')
	print(f'{"Fr":10}{Fr.item():^10.4f}')
	print(f'{"U" + prime:10}{U.item():^10.4f}')
	print(f'{"periods" + prime:10}{num_periods:^10}')
	print(f'{delta_t_char + prime:10}{delta_t:^10.4g}')
	print(f'{"Re_p":10}{general_Re_p:^10.4g}\n')

def write_data(results, filepath):
	"""Writes the provided `results` dictionary to the provided csv file."""
	df = pd.DataFrame(results)
	numerics = df.explode(list(df.columns.values), ignore_index=True)
	path_list = filepath.split('/')
	filename = path_list[-1]
	if os.path.exists(filepath): # append if file already exists
		numerics.to_csv(filepath, mode='a', header=False,
						index=False)
		print(f'Data appended to {filename}.')
	else:
		numerics.to_csv(filepath, index=False)
		print(f'Data written to {filename}.')

def plot_trajectory(results1, results2):
	"""Plots the trajectory of the particle with and without history effects."""
	x1, z1 = results1['x'], results1['z']
	x2, z2 = results2['x'], results2['z']
	x_crossings1, z_crossings1, _, _, _ = ts.compute_drift_velocity(x1, z1,
											 results1['xdot'], results1['t'])
	x_crossings2, z_crossings2, _, _, _ = ts.compute_drift_velocity(x2, z2,
											 results2['xdot'], results2['t'])
	print('Plotting...')
	fs, lfs = 14, 16 # font sizes
	plt.figure()
	plt.xlabel('x', fontsize=lfs)
	plt.ylabel('z', fontsize=lfs)
	plt.xticks(fontsize=fs)
	plt.yticks(fontsize=fs)
	plt.minorticks_on()
	plt.plot(x1, z1, '-k', label='without history')
	plt.plot(x2, z2, ':k', label='with history')
	plt.scatter(x_crossings1, z_crossings1, c='k', marker='x')
	plt.scatter(x_crossings2, z_crossings2, c='k', marker='x')
	plt.legend(fontsize=fs)
	plt.show()

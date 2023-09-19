import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np
from time import time
from tqdm import tqdm
from itertools import chain

from transport_framework import particle as prt 
from models import deep_water_wave as fl
from models import haller_system as h
from models import my_system as my

def main():
	"""
	This program computes the global error between the numerical solutions of an
	inertial particle in a deep water wave as produced by the Haller & Sapsis
	(2008) method and the Daitche (2013) method, and saves the results to the
	`data` directory.
	"""
	# initialize delta_t values and dictionaries to store numerical solutions
	timesteps = np.linspace(1e-3, 5e-2, 10)
	labels = list(chain.from_iterable(('x_%.2e' % delta_t, 'z_%.2e' % delta_t)
				  for delta_t in timesteps))
	dict1 = dict.fromkeys(labels)
	dict2 = dict.fromkeys(labels)
	dict3 = dict.fromkeys(labels)
	haller_dict = dict.fromkeys(labels)
	time_dict = dict.fromkeys(['delta_t', 'computation_time'])
	time_dict['delta_t'] = timesteps
	computation_times = []

	# initialize variables for numerical simulations
	num_periods = 20
	R = 2 / 3 * 0.98
	my_flow = fl.DeepWaterWave(amplitude=0.026, wavelength=0.5)
	my_particle = prt.Particle(stokes_num=my_flow.froude_num * R * 0.5)
	my_system = my.MyTransportSystem(my_particle, my_flow, R)
	haller_system = h.HallerTransportSystem(my_particle, my_flow, R)
	x_0, z_0 = 0, 0 
	xdot_0, zdot_0 = my_flow.velocity(x_0, z_0, 0)

	# compute and store first, second, and third order numerical solutions
	for delta_t in timesteps:
		print(f'Computing numerics for delta_t = %.2e...' % delta_t)
		start = time()
		x_label = 'x_%.2e' % delta_t
		z_label = 'z_%.2e' % delta_t
		x1, z1, _, _, _ = my_system.run_numerics(include_history=False, order=1,
												 x_0=x_0, z_0=z_0,
												 xdot_0=xdot_0, zdot_0=zdot_0,
												 delta_t=delta_t,
												 num_periods=num_periods)
		x2, z2, _, _, _ = my_system.run_numerics(include_history=False, order=2,
												 x_0=x_0, z_0=z_0,
												 xdot_0=xdot_0, zdot_0=zdot_0,
												 delta_t=delta_t,
												 num_periods=num_periods)
		x3, z3, _, _, _ = my_system.run_numerics(include_history=False, x_0=x_0,
												 z_0=z_0, xdot_0=xdot_0,
												 zdot_0=zdot_0, delta_t=delta_t,
												 num_periods=num_periods)
		xh, zh, _, _, _ = haller_system.run_numerics(haller_system.maxey_riley,
													 x_0=x_0, z_0=z_0,
													 delta_t=delta_t,
													 num_periods=num_periods)
		finish = time()
		dict1[x_label] = x1
		dict1[z_label] = z1
		dict2[x_label] = x2
		dict2[z_label] = z2
		dict3[x_label] = x3
		dict3[z_label] = z3
		haller_dict[x_label] = xh
		haller_dict[z_label] = zh
		computation_times.append(finish - start)
		print(f'Computations for delta_t = %.2e complete.\t\t%5.2fs\n'
			  % (delta_t, finish - start))

	# insert NaNs to fix ragged data
	dict1 = dict([(key, pd.Series(value)) for key, value in dict1.items()])
	dict2 = dict([(key, pd.Series(value)) for key, value in dict2.items()])
	dict3 = dict([(key, pd.Series(value)) for key, value in dict3.items()])
	haller_dict = dict([(key, pd.Series(value)) for key, value
						in haller_dict.items()])
	numerics1 = pd.DataFrame(dict1)
	numerics2 = pd.DataFrame(dict2)
	numerics3 = pd.DataFrame(dict3)
	haller_numerics = pd.DataFrame(haller_dict)

	# compute global error
	print('Computing global error...')
	global_error1, global_error2, global_error3 = [], [], []
	for delta_t in tqdm(timesteps):
		x_label = 'x_%.2e' % delta_t
		z_label = 'z_%.2e' % delta_t
		haller = haller_numerics[[x_label, z_label]].dropna().to_numpy()
		num1 = numerics1[[x_label, z_label]].dropna().to_numpy()
		num2 = numerics2[[x_label, z_label]].dropna().to_numpy()
		num3 = numerics3[[x_label, z_label]].dropna().to_numpy()
		global_error1.append(np.linalg.norm(haller - num1, axis=1).max())
		global_error2.append(np.linalg.norm(haller - num2, axis=1).max())
		global_error3.append(np.linalg.norm(haller - num3, axis=1).max())

	# write solutions to data files
	global_error_dict = {'delta_t': timesteps, 'global_error1': global_error1,
											   'global_error2': global_error2,
											   'global_error3': global_error3}
	global_error = pd.DataFrame(global_error_dict)
	global_error.to_csv('../data/deep_water_wave/global_error.csv', index=False)
	time_dict['computation_time'] = computation_times
	times = pd.DataFrame(time_dict)
	times.to_csv('../data/deep_water_wave/computation_times.csv', index=False)

if __name__ == '__main__':
	main()

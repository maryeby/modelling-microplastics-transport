import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np
from time import time
from tqdm import tqdm
from itertools import chain

from transport_framework import particle as prt 
from models import rotating_flow as fl
from models import rotating_system as ts

def main():
	"""
	This program computes the global error for the numerical solutions of a
	rotating particle in a flow and saves the results to the `data` directory.
	"""
	# read analytical data
	analytics = pd.read_csv('../data/rotating_analytics_varying_timesteps.csv')

	# initialize delta_t values and dictionaries to store numerical solutions
	timesteps = np.linspace(1e-3, 1e-1, 10)
	labels = list(chain.from_iterable(('x_%.2e' % delta_t, 'z_%.2e' % delta_t)
				  for delta_t in timesteps))
	dict1 = dict.fromkeys(labels)
	dict2 = dict.fromkeys(labels)
	dict3 = dict.fromkeys(labels)
	time_dict = dict.fromkeys(['delta_t', 'computation_time'])
	time_dict['delta_t'] = timesteps
	computation_times = []

	# initialize variables for numerical simulations
	t_final = 10
	R = 2 / 3 * 0.75
	my_particle = prt.Particle(stokes_num=2 / 3 * 0.3)
	my_flow = fl.RotatingFlow()
	my_system = ts.RotatingTransportSystem(my_particle, my_flow, R)
	x_0 = 1 
	xdot_0, zdot_0 = my_flow.velocity(x_0, 0)

	# compute and store first, second, and third order numerical solutions
	for delta_t in timesteps:
		print(f'Computing numerics for delta_t = %.2e...' % delta_t)
		start = time()
		x_label = 'x_%.2e' % delta_t
		z_label = 'z_%.2e' % delta_t
		t = np.arange(0, t_final + delta_t, delta_t)
		x1, z1, _, _, _ = my_system.run_numerics(include_history=True, order=1,
												 x_0=x_0, xdot_0=xdot_0,
												 zdot_0=zdot_0, delta_t=delta_t,
												 num_periods=t_final)
		x2, z2, _, _, _ = my_system.run_numerics(include_history=True, order=2,
												 x_0=x_0, xdot_0=xdot_0,
												 zdot_0=zdot_0, delta_t=delta_t,
												 num_periods=t_final)
		x3, z3, _, _, _ = my_system.run_numerics(include_history=True, x_0=x_0,
												 xdot_0=xdot_0, zdot_0=zdot_0,
												 delta_t=delta_t,
												 num_periods=t_final)
		finish = time()
		dict1[x_label] = x1
		dict1[z_label] = z1
		dict2[x_label] = x2
		dict2[z_label] = z2
		dict3[x_label] = x3
		dict3[z_label] = z3
		computation_times.append(finish - start)
		print(f'Computations for delta_t = %.2e complete.\t\t%5.2fs\n'
			  % (delta_t, finish - start))

	# insert NaNs to fix ragged data
	dict1 = dict([(key, pd.Series(value)) for key, value in dict1.items()])
	dict2 = dict([(key, pd.Series(value)) for key, value in dict2.items()])
	dict3 = dict([(key, pd.Series(value)) for key, value in dict3.items()])
	numerics1 = pd.DataFrame(dict1)
	numerics2 = pd.DataFrame(dict2)
	numerics3 = pd.DataFrame(dict3)

	# compute global error
	print('Computing global error...')
	global_error1, global_error2, global_error3 = [], [], []
	for delta_t in tqdm(timesteps):
		x_label = 'x_%.2e' % delta_t
		z_label = 'z_%.2e' % delta_t
		exact = analytics[[x_label, z_label]].dropna().to_numpy()
		num1 = numerics1[[x_label, z_label]].dropna().to_numpy()
		num2 = numerics2[[x_label, z_label]].dropna().to_numpy()
		num3 = numerics3[[x_label, z_label]].dropna().to_numpy()
		global_error1.append(np.linalg.norm(exact - num1, axis=1).max())
		global_error2.append(np.linalg.norm(exact - num2, axis=1).max())
		global_error3.append(np.linalg.norm(exact - num3, axis=1).max())

	# write solutions to data files
	global_error_dict = {'delta_t': timesteps, 'global_error1': global_error1,
											   'global_error2': global_error2,
											   'global_error3': global_error3}
	global_error = pd.DataFrame(global_error_dict)
	global_error.to_csv('../data/rotating_global_error.csv', index=False)
	time_dict['computation_time'] = computation_times
	times = pd.DataFrame(time_dict)
	times.to_csv('../data/rotating_computation_times.csv', index=False)

if __name__ == '__main__':
	main()

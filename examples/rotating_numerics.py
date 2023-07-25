import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
import pandas as pd

from transport_framework import particle as prt
from models import rotating_flow as fl
from models import rotating_system as ts

def main():
	"""
	This program runs numerical simulations for a rotating rigid body, and saves
	the results to the `data` directory.
	"""
	# initialize variables for the transport system
	t_final = 20
	R = 2 / 3 * 0.75
	my_particle = prt.Particle(stokes_num=2 / 3 * 0.3)
	my_flow = fl.RotatingFlow()
	my_system = ts.RotatingTransportSystem(my_particle, my_flow, R)
	my_dict = dict.fromkeys(['t', 'first_x', 'first_z', 'first_xdot',
							 'first_zdot', 'second_x', 'second_z',
							 'second_xdot', 'second_zdot', 'third_x', 'third_z',
							 'third_xdot', 'third_zdot'])
	x_0 = 1
	xdot_0, zdot_0 = my_flow.velocity(x_0, 0)

	# compute first order results
	x, z, xdot, zdot, _ = my_system.run_numerics(include_history=True, order=1,
												 x_0=x_0, xdot_0=xdot_0,
												 zdot_0=zdot_0, delta_t=1e-2,
												 num_periods=t_final)
	my_dict['first_x'] = x
	my_dict['first_z'] = z
	my_dict['first_xdot'] = xdot
	my_dict['first_zdot'] = zdot

	# compute second order results
	x, z, xdot, zdot, _ = my_system.run_numerics(include_history=True, order=2,
												 x_0=x_0, xdot_0=xdot_0,
												 zdot_0=zdot_0, delta_t=1e-2,
												 num_periods=t_final)
	my_dict['second_x'] = x
	my_dict['second_z'] = z
	my_dict['second_xdot'] = xdot
	my_dict['second_zdot'] = zdot

	# compute third order results
	x, z, xdot, zdot, t = my_system.run_numerics(include_history=True, x_0=x_0,
												 xdot_0=xdot_0, zdot_0=zdot_0,
												 delta_t=1e-2,
												 num_periods=t_final)
	my_dict['third_x'] = x
	my_dict['third_z'] = z
	my_dict['third_xdot'] = xdot
	my_dict['third_zdot'] = zdot

	# get all times t and the position at integer times t_int
	my_dict['t'] = t
	int_indices = np.where(t == t.astype(int))
	int_x = np.take(x, int_indices[0])
	int_z = np.take(z, int_indices[0])
	time_dict = {'int_x': int_x, 'int_z': int_z}

	# write results to data file
	numerics = pd.DataFrame(my_dict)
	int_times = pd.DataFrame(time_dict)
	numerics.to_csv('data/rotating_numerics.csv', index=False)
	int_times.to_csv('data/rotating_int_times.csv', index=False)
		
if __name__ == '__main__':
	main()

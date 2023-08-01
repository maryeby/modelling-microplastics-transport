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
	t_final = 100
	delta_t = 1e-2
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
	x, z, _, _, _ = my_system.run_numerics(include_history=True, order=1,
										   x_0=x_0, xdot_0=xdot_0,
										   zdot_0=zdot_0, delta_t=delta_t,
										   num_periods=t_final)
	my_dict['first_x'] = x
	my_dict['first_z'] = z

	# compute second order results
	x, z, _, _, _ = my_system.run_numerics(include_history=True, order=2,
										   x_0=x_0, xdot_0=xdot_0,
										   zdot_0=zdot_0, delta_t=delta_t,
										   num_periods=t_final)
	my_dict['second_x'] = x
	my_dict['second_z'] = z

	# compute third order results
	x, z, _, _, t = my_system.run_numerics(include_history=True, x_0=x_0,
										   xdot_0=xdot_0, zdot_0=zdot_0,
										   delta_t=delta_t, num_periods=t_final)
	my_dict['third_x'] = x
	my_dict['third_z'] = z
	my_dict['t'] = t

	# write results to data file
	numerics = pd.DataFrame(my_dict)
	numerics.to_csv('../data/rotating_numerics.csv', index=False)
		
if __name__ == '__main__':
	main()

import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
import pandas as pd

from transport_framework import particle as prt
from models import rotating_flow as fl
from models import my_system as ts

def main():
	"""
	This program runs numerical simulations for a rotating rigid body, and saves
	the results to the `data/rigid_body_rotation` directory.
	"""
	# initialize variables for the transport system
	R = 2 / 3 * 0.75
	my_particle = prt.Particle(stokes_num=2 / 3 * 0.3)
	my_flow = fl.RotatingFlow()
	my_system = ts.MyTransportSystem(my_particle, my_flow, R)

	# initialize variables for the numerical simulations
	x_0, z_0 = 1, 0
	xdot_0, zdot_0 = my_flow.velocity(x_0, 0)
	t_final = 10
	delta_t = 1e-2
	include_history = True

	# run simulation and store results in dictionary
	x, z, xdot, zdot, t, _, _, _, _, _, _, _, _, H_x, H_z, \
	   history_x, history_z = my_system.run_numerics(include_history, x_0, z_0,
													 xdot_0, zdot_0,
													 t_final, delta_t,
													 hide_progress=False,
													 include_forces=True)
	u_x, u_z = my_flow.velocity(x, z, t)
	w_x, w_z = xdot - u_x, zdot - u_z
	# write results to data file
	numerics = pd.DataFrame({'t': t, 'x': x, 'z': z, 'v_x': xdot, 'v_z': zdot,
							 'u_x': u_x, 'u_z': u_z, 'w_x': w_x, 'w_z': w_z,
							 'H_x': H_x, 'H_z': H_z,
							 'H\'_x': history_x, 'H\'_z': history_z})
	numerics.to_csv('../../data/rigid_body_rotation/numerical_history.csv',
					 index=False)
		
if __name__ == '__main__':
	main()

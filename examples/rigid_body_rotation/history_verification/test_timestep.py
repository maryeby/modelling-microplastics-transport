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
	t_final = 3
	delta_ts = [1e-2, 5e-3, 1e-3]
	include_history = True
	history_x0s, history_z0s = [], []

	for delta_t in delta_ts:
		# run simulation and store results
		print('delta_t:', delta_t)
		_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, \
		   history_x, history_z = my_system.run_numerics(include_history,
														 x_0, z_0,
														 xdot_0, zdot_0,
														 t_final, delta_t,
														 hide_progress=False,
														 include_forces=True)
		history_x0s.append(history_x[0])
		history_z0s.append(history_z[0])
		
	# write results to data file
	numerics = pd.DataFrame({'delta_t': delta_ts, 'H\'(0)_x': history_x0s,
							 'H\'(0)_z': history_z0s})
	numerics.to_csv('../../data/rigid_body_rotation/timestep_test.csv',
					 index=False)
		
if __name__ == '__main__':
	main()

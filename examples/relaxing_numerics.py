import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd

from transport_framework import particle as prt
from models import quiescent_flow as fl
from models import relaxing_system as ts

def main():
	"""
	This program runs numerical simulations for a relaxing particle in a
	quiescent flow, and saves the results to the `data` directory.
	"""
	# initialize variables for the transport system
	t_final = 15
	R_light = 2 / (1 + 2 * 0.01)
	R_neutral = 2 / (1 + 2 * 1)
	R_heavy = 2 / (1 + 2 * 5)
	my_particle = prt.Particle(stokes_num=0.66)
	my_flow = fl.QuiescentFlow()
	light_system = ts.RelaxingTransportSystem(my_particle, my_flow, R_light)
	neutral_system = ts.RelaxingTransportSystem(my_particle, my_flow, R_neutral)
	heavy_system = ts.RelaxingTransportSystem(my_particle, my_flow, R_heavy)
	my_dict = dict.fromkeys(['t', 'light_x_history', 'light_z_history',
							 'light_xdot_history', 'light_zdot_history',
							 'neutral_x_history', 'neutral_z_history',
							 'neutral_xdot_history', 'neutral_zdot_history',
							 'heavy_x_history', 'heavy_z_history',
							 'heavy_xdot_history', 'heavy_zdot_history',
							 'light_x', 'light_z', 'light_xdot', 'light_zdot',
							 'neutral_x', 'neutral_z', 'neutral_xdot',
							 'neutral_zdot', 'heavy_x', 'heavy_z',
							 'heavy_xdot', 'heavy_zdot'])

	# compute results for the light particle
	x, z, xdot, zdot, t = light_system.run_numerics(include_history=True,
													num_periods=t_final)
	my_dict['t'] = t
	my_dict['light_x_history'] = x
	my_dict['light_z_history'] = z
	my_dict['light_xdot_history'] = xdot
	my_dict['light_zdot_history'] = zdot
	x, z, xdot, zdot, t = light_system.run_numerics(include_history=False,
													num_periods=t_final)
	my_dict['light_x'] = x
	my_dict['light_z'] = z
	my_dict['light_xdot'] = xdot
	my_dict['light_zdot'] = zdot
	x, z, xdot, zdot, t = neutral_system.run_numerics(include_history=True,
													  num_periods=t_final)
	my_dict['neutral_x_history'] = x
	my_dict['neutral_z_history'] = z
	my_dict['neutral_xdot_history'] = xdot
	my_dict['neutral_zdot_history'] = zdot
	x, z, xdot, zdot, t = neutral_system.run_numerics(include_history=False,
													  num_periods=t_final)
	my_dict['neutral_x'] = x
	my_dict['neutral_z'] = z
	my_dict['neutral_xdot'] = xdot
	my_dict['neutral_zdot'] = zdot
	x, z, xdot, zdot, t = heavy_system.run_numerics(include_history=True,
													num_periods=t_final)
	my_dict['heavy_x_history'] = x
	my_dict['heavy_z_history'] = z
	my_dict['heavy_xdot_history'] = xdot
	my_dict['heavy_zdot_history'] = zdot
	x, z, xdot, zdot, t = heavy_system.run_numerics(include_history=False,
													num_periods=t_final)
	my_dict['heavy_x'] = x
	my_dict['heavy_z'] = z
	my_dict['heavy_xdot'] = xdot
	my_dict['heavy_zdot'] = zdot

	# write results to data file
	numerics = pd.DataFrame(my_dict)
	numerics.to_csv('data/relaxing_numerics.csv', index=False)
		
if __name__ == '__main__':
	main()

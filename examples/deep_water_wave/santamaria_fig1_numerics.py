import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
import pandas as pd

from transport_framework import particle as prt
from models import dim_deep_water_wave as dim_fl
from models import deep_water_wave as fl
from models import santamaria_system as sm
from models import my_system as my

def main():
	"""
	This program reproduces numerical results from Figure 1 in Santamaria et al.
	(2013) and saves the results to the `data/deep_water_wave` directory.
	"""

	# initialize variables for the Santamaria system
	heavy_beta = 0.96
	light_beta = 1.04
	sm_flow = dim_fl.DimensionalDeepWaterWave(amplitude=0.026, wavelength=0.5)
	sm_heavy_particle = prt.Particle(stokes_num=0.5)
	sm_light_particle = prt.Particle(stokes_num=0.5)
	sm_heavy_system = sm.SantamariaTransportSystem(sm_heavy_particle, sm_flow,
												   heavy_beta)
	sm_light_system = sm.SantamariaTransportSystem(sm_light_particle, sm_flow,
												   light_beta)
	k = sm_flow.wavenum

	# initialize variables for the Daitche system
	heavy_R = 2 / 3 * 0.96
	light_R = 2 / 3 * 1.04
	my_flow = fl.DeepWaterWave(amplitude=0.026, wavelength=0.5)
	heavy_st = my_flow.froude_num * heavy_R * 0.5
	light_st = my_flow.froude_num * light_R * 0.5
	my_heavy_particle = prt.Particle(stokes_num=heavy_st)
	my_light_particle = prt.Particle(stokes_num=light_st)
	my_heavy_system = my.MyTransportSystem(my_heavy_particle, my_flow, heavy_R)
	my_light_system = my.MyTransportSystem(my_light_particle, my_flow, light_R)

	# initialize variables for the numerical simulations
	my_dict = dict.fromkeys(['sm_heavy_x', 'sm_heavy_z', 'sm_light_x',
							 'sm_light_z', 'my_heavy_x', 'my_heavy_z',
							 'my_light_x', 'my_light_z'])
	x_0, z_0 = k * 0.13, k * -0.4
	T = 1 / (sm_flow.angular_freq * sm_flow.froude_num)
	num_periods = 50
	delta_t = 1e-2

	# generate results for the heavy and light particles
	x, z, _, _, _ = sm_heavy_system.run_numerics(sm_heavy_system.maxey_riley,
												 x_0=0, z_0=0,
												 delta_t=delta_t,
												 num_periods=num_periods)
	my_dict['sm_heavy_x'] = k * x
	my_dict['sm_heavy_z'] = k * z
	x, z, _, _, _ = sm_light_system.run_numerics(sm_light_system.maxey_riley,
												 x_0=0.13, z_0=-0.4,
												 delta_t=delta_t,
												 num_periods=num_periods)
	my_dict['sm_light_x'] = k * x
	my_dict['sm_light_z'] = k * z
	x, z, _, _, _ = my_heavy_system.run_numerics(include_history=False, x_0=0,
												 z_0=0, xdot_0=1, zdot_0=0,
												 delta_t=delta_t / T,
												 num_periods=num_periods / T)
	my_dict['my_heavy_x'] = x
	my_dict['my_heavy_z'] = z
	x, z, _, _, _ = my_light_system.run_numerics(include_history=False, x_0=x_0,
												 z_0=z_0, xdot_0=0, zdot_0=0,
												 delta_t=delta_t / T,
                                                 num_periods=num_periods / T)
	my_dict['my_light_x'] = x
	my_dict['my_light_z'] = z

	# write results to data file
	my_dict = dict([(key, pd.Series(value)) for key, value in my_dict.items()])
	numerics = pd.DataFrame(my_dict)
	numerics.to_csv('../data/deep_water_wave/santamaria_fig1_recreation.csv',
					index=False)
if __name__ == '__main__':
	main()

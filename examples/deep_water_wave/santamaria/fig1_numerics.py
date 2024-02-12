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
	A = 0.026		# amplitude
	wavelength = 0.5
	sm_stokes_num = 0.5
	sm_flow = dim_fl.DimensionalDeepWaterWave(amplitude=A,
											  wavelength=wavelength)
	sm_heavy_particle = prt.Particle(stokes_num=sm_stokes_num)
	sm_light_particle = prt.Particle(stokes_num=sm_stokes_num)
	sm_heavy_system = sm.SantamariaTransportSystem(sm_heavy_particle, sm_flow,
												   heavy_beta)
	sm_light_system = sm.SantamariaTransportSystem(sm_light_particle, sm_flow,
												   light_beta)
	# initialize variables to use for scaling
	k = sm_flow.wavenum
	Fr = sm_flow.froude_num
	omega = sm_flow.angular_freq
	T = omega * Fr # time scaling

	# initialize variables for the Daitche system
	heavy_R = 2 / 3 * heavy_beta
	light_R = 2 / 3 * light_beta
	my_flow = fl.DeepWaterWave(amplitude=A, wavelength=wavelength)
	heavy_st = Fr * heavy_R * sm_stokes_num
	light_st = Fr * light_R * sm_stokes_num
	my_heavy_particle = prt.Particle(stokes_num=heavy_st)
	my_light_particle = prt.Particle(stokes_num=light_st)
	my_heavy_system = my.MyTransportSystem(my_heavy_particle, my_flow, heavy_R)
	my_light_system = my.MyTransportSystem(my_light_particle, my_flow, light_R)

	# initialize variables for the numerical simulations
	my_dict = {}
	num_periods = 200
	delta_t = 5e-3
	heavy_x_0, heavy_z_0, light_x_0, light_z_0 = 0, 0, 0.13, -0.4
	xdot_0_light, zdot_0_light = my_light_system.flow.velocity(k * light_x_0,
															   k * light_z_0,
															   t=0)
	xdot_0_heavy, zdot_0_heavy = my_heavy_system.flow.velocity(k * heavy_x_0,
															   k * heavy_z_0,
															   t=0)
	# generate results for the heavy and light particles
	x, z, _, _, _ = sm_heavy_system.run_numerics(sm_heavy_system.maxey_riley,
												 heavy_x_0, heavy_z_0,
												 num_periods / T, delta_t / T)
	my_dict['sm_heavy_x'] = k * x
	my_dict['sm_heavy_z'] = k * z
	x, z, _, _, _ = sm_light_system.run_numerics(sm_light_system.maxey_riley,
												 light_x_0, light_z_0,
												 num_periods / T, delta_t / T)
	my_dict['sm_light_x'] = k * x
	my_dict['sm_light_z'] = k * z
	x, z, _, _, _ = my_heavy_system.run_numerics(include_history=False,
												 x_0=k * heavy_x_0,
												 z_0=k * heavy_z_0,
												 xdot_0=xdot_0_heavy,
												 zdot_0=zdot_0_heavy,
												 delta_t=delta_t * T,
												 num_periods=num_periods * T,
												 hide_progress=True)
	my_dict['my_heavy_x'] = x
	my_dict['my_heavy_z'] = z
	x, z, _, _, _ = my_light_system.run_numerics(include_history=False,
												 x_0=k * light_x_0,
												 z_0=k * light_z_0,
												 xdot_0=xdot_0_light,
												 zdot_0=zdot_0_light,
												 delta_t=delta_t * T,
                                                 num_periods=num_periods * T,
												 hide_progress=True)
	my_dict['my_light_x'] = x
	my_dict['my_light_z'] = z

	# write results to data file
	my_dict = dict([(key, pd.Series(value)) for key, value in my_dict.items()])
	numerics = pd.DataFrame(my_dict)
	numerics.to_csv('../../data/deep_water_wave/santamaria_fig1_recreation.csv',
					index=False)
if __name__ == '__main__':
	main()

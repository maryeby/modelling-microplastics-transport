import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
import pandas as pd

from transport_framework import particle as prt
from models import dim_deep_water_wave as dim_fl
from models import deep_water_wave as fl
from models import santamaria_system as sm
from models import my_system as ts

def main():
	"""
	This program reproduces numerical results from Figure 2 in Santamaria et al.
	(2013).
	"""
	# initialize the particle, flow, and transport system
	my_flow = fl.DeepWaterWave(amplitude=0.02, wavelength=1)
	R = 2 / 3 * 0.9
	St = my_flow.froude_num * R * 0.157
	my_particle = prt.Particle(stokes_num=St)
	my_system = ts.MyTransportSystem(my_particle, my_flow, R)
	my_dict = dict.fromkeys(['coarse_t', 'coarse_u_d', 'coarse_w_d', 'medium_t',
							 'medium_u_d', 'medium_w_d', 'fine_t', 'fine_u_d',
							 'fine_w_d'])

	# intialize Santamaria particle, flow, transport system
	sm_flow = dim_fl.DimensionalDeepWaterWave(amplitude=0.02, wavelength=1)
	sm_particle = prt.Particle(stokes_num=0.157)
	sm_system = sm.SantamariaTransportSystem(sm_particle, sm_flow, 0.9)
	omega = sm_flow.angular_freq
	U = sm_flow.max_velocity
	T = 1 / (omega * sm_flow.froude_num)
	num_periods = 50
	x_0 = 0
	z_0 = 0
	xdot_0, zdot_0 = my_flow.velocity(x_0, z_0, t=0)

	# run numerical simulations and compute drift velocity for Santamaria
	x, z, xdot, _, t = sm_system.run_numerics(sm_system.maxey_riley,
											  x_0=x_0, z_0=z_0, delta_t=1e-3,
											  num_periods=num_periods)
	sm_u_d, sm_w_d, sm_t = compute_drift_velocity(x, z, xdot, t)
	my_dict['sm_u_d'] = sm_u_d / U
	my_dict['sm_w_d'] = sm_w_d / U
	my_dict['sm_t'] = omega * sm_t

	# run numerical simulation and compute drift velocity for fine delta_t
	x, z, xdot, _, t = my_system.run_numerics(include_history=False,
											  x_0=x_0, z_0=z_0,
											  xdot_0=xdot_0, zdot_0=zdot_0,
											  delta_t=1e-3 / T,
											  num_periods=num_periods / T)
	fine_u_d, fine_w_d, fine_t = compute_drift_velocity(x, z, xdot, t)
	my_dict['fine_u_d'] = fine_u_d
	my_dict['fine_w_d'] = fine_w_d
	my_dict['fine_t'] = fine_t * omega

	# run numerical simulation and compute drift velocity for medium delta_t
	x, z, xdot, _, t = my_system.run_numerics(include_history=False,
											  x_0=x_0, z_0=z_0,
											  xdot_0=xdot_0, zdot_0=zdot_0,
											  delta_t=5e-3 / T,
											  num_periods=num_periods / T)
	medium_u_d, medium_w_d, medium_t = compute_drift_velocity(x, z, xdot, t)
	my_dict['medium_u_d'] = medium_u_d
	my_dict['medium_w_d'] = medium_w_d
	my_dict['medium_t'] = medium_t * omega

	# run numerical simulation and compute drift velocity for coarse delta_t
	x, z, xdot, _, t = my_system.run_numerics(include_history=False,
											  x_0=x_0, z_0=z_0,
											  xdot_0=xdot_0, zdot_0=zdot_0,
											  delta_t=1e-2 / T,
											  num_periods=num_periods / T)
	coarse_u_d, coarse_w_d, coarse_t = compute_drift_velocity(x, z, xdot, t)
	my_dict['coarse_u_d'] = coarse_u_d
	my_dict['coarse_w_d'] = coarse_w_d
	my_dict['coarse_t'] = coarse_t * omega

	# write results to data file
	my_dict = dict([(key, pd.Series(value)) for key, value in my_dict.items()])
	numerics = pd.DataFrame(my_dict)
	numerics.to_csv('../data/deep_water_wave/santamaria_fig2_recreation.csv',
					index=False)

def compute_drift_velocity(x, z, xdot, t):
	# find estimated endpoints of periods
	estimated_endpoints = []
	for i in range(1, len(xdot)):
		if xdot[i - 1] < 0 and 0 <= xdot[i]:
			estimated_endpoints.append(i)
	
	# find exact endpoints of periods using interpolation
	interpd_x, interpd_z, interpd_t = [], [], []
	for i in range(1, len(estimated_endpoints)):
		current = estimated_endpoints[i]
		previous = current - 1

		new_t = np.interp(0, [xdot[previous], xdot[current]], [t[previous],
															   t[current]])
		interpd_t.append(new_t)
		interpd_x.append(np.interp(new_t, [t[previous], t[current]],
								   [x[previous], x[current]]))
		interpd_z.append(np.interp(new_t, [t[previous], t[current]],
								   [z[previous], z[current]]))

	# compute drift velocity
	u_d, w_d = [], []
	for i in range(1, len(interpd_t)):
		u_d.append((interpd_x[i] - interpd_x[i - 1]) 
				 / (interpd_t[i] - interpd_t[i - 1]))
		w_d.append((interpd_z[i] - interpd_z[i - 1]) 
				 / (interpd_t[i] - interpd_t[i - 1]))
	return np.array(u_d), np.array(w_d), np.array(interpd_t)

if __name__ == '__main__':
	main()

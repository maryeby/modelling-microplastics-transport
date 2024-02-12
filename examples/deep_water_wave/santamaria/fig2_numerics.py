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
	(2013) and saves the results to the `data/deep_water_wave` directory.
	"""
	# intialize Santamaria particle, flow, transport system
	wavelength = 1
	A = 0.02	# amplitude
	beta = 0.9	# heavy particle
	sm_stokes_num = 0.157
	sm_flow = dim_fl.DimensionalDeepWaterWave(amplitude=A,
											  wavelength=wavelength)
	sm_particle = prt.Particle(stokes_num=sm_stokes_num)
	sm_system = sm.SantamariaTransportSystem(sm_particle, sm_flow, beta)

	# initialize parameters for scaling
	U = sm_flow.max_velocity
	k = sm_flow.wavenum
	omega = sm_flow.angular_freq
	Fr = sm_flow.froude_num
	T = omega * Fr	# time scaling

	# initialize the particle, flow, and transport system
	my_flow = fl.DeepWaterWave(amplitude=A, wavelength=wavelength)
	R = 2 / 3 * beta
	my_St = my_flow.froude_num * R * sm_stokes_num
	my_particle = prt.Particle(stokes_num=my_St)
	my_system = ts.MyTransportSystem(my_particle, my_flow, R)

	# initialize parameters for the numerical simulations
	my_dict = {}
	num_periods = 50
	x_0, z_0 = 0, 0
	xdot_0, zdot_0 = my_flow.velocity(k * x_0, k * z_0, t=0)

	# run numerical simulations and compute drift velocity for Santamaria
	x, z, xdot, _, t = sm_system.run_numerics(sm_system.maxey_riley,
											  x_0, z_0,
											  num_periods / T, delta_t=1e-3 / T)
	sm_u_d, sm_w_d, sm_t = compute_drift_velocity(x, z, xdot, t)
	my_dict['sm_u_d'] = sm_u_d / U
	my_dict['sm_w_d'] = sm_w_d / U
	my_dict['sm_t'] = omega * sm_t

	# run numerical simulation and compute drift velocity for fine delta_t
	x, z, xdot, _, t = my_system.run_numerics(include_history=False,
											  x_0=k * x_0, z_0=k * z_0,
											  xdot_0=xdot_0, zdot_0=zdot_0,
											  delta_t=1e-3 * T,
											  num_periods=num_periods * T,
											  hide_progress=False)
	fine_u_d, fine_w_d, fine_t = compute_drift_velocity(x, z, xdot, t)
	my_dict['fine_u_d'] = fine_u_d
	my_dict['fine_w_d'] = fine_w_d
	my_dict['fine_t'] = fine_t / Fr

	# run numerical simulation and compute drift velocity for medium delta_t
	x, z, xdot, _, t = my_system.run_numerics(include_history=False,
											  x_0=k * x_0, z_0=k * z_0,
											  xdot_0=xdot_0, zdot_0=zdot_0,
											  delta_t=5e-3 * T,
											  num_periods=num_periods * T,
											  hide_progress=False)
	medium_u_d, medium_w_d, medium_t = compute_drift_velocity(x, z, xdot, t)
	my_dict['medium_u_d'] = medium_u_d
	my_dict['medium_w_d'] = medium_w_d
	my_dict['medium_t'] = medium_t / Fr

	# run numerical simulation and compute drift velocity for coarse delta_t
	x, z, xdot, _, t = my_system.run_numerics(include_history=False,
											  x_0=k * x_0, z_0=k * z_0,
											  xdot_0=xdot_0, zdot_0=zdot_0,
											  delta_t=1e-2 * T,
											  num_periods=num_periods * T,
											  hide_progress=False)
	coarse_u_d, coarse_w_d, coarse_t = compute_drift_velocity(x, z, xdot, t)
	my_dict['coarse_u_d'] = coarse_u_d
	my_dict['coarse_w_d'] = coarse_w_d
	my_dict['coarse_t'] = coarse_t / Fr

	# write results to data file
	my_dict = dict([(key, pd.Series(value)) for key, value in my_dict.items()])
	numerics = pd.DataFrame(my_dict)
	numerics.to_csv('../../data/deep_water_wave/santamaria_fig2_recreation.csv',
					index=False)

def compute_drift_velocity(x, z, xdot, t):
	r"""
	Computes the Stokes drift velocity
	$$\mathbf{u}_d = \langle u_d, w_d \rangle$$
	using the distance travelled by the particle averaged over each wave period,
	$$\mathbf{u}_d = \frac{\mathbf{x}_{n + 1} - \mathbf{x}_n}{\text{period}}.$$
	"""
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

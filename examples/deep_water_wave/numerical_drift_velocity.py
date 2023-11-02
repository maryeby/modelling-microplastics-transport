import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
import pandas as pd
import scipy.constants as constants

from transport_framework import particle as prt
from models import deep_water_wave as fl
from models import my_system as ts

def main():
	r"""
	This program numerically computes the Stokes drift velocity of an inertial
	particle in linear deep water waves for various Stokes numbers, and saves
	the results to the `data/deep_water_wave` directory.
	"""
	# initialize parameters for the flow and particle
	stokes_nums = [0.01, 0.1, 1, 10]
	depth = 10
	amplitude = 0.01
	wavelength = 2
	initial_depths = np.linspace(0, -depth, 10, endpoint=False)

	# store parameters
	my_dict = {}
	my_dict['St'] = stokes_nums
	my_dict['h'] = depth
	my_dict['z'] = initial_depths
	my_dict['z/h'] = initial_depths / depth
	my_dict['A'] = amplitude
	my_dict['lambda'] = wavelength

	# initialize parameters for the transport system and numerics
	my_wave = fl.DeepWaterWave(depth=depth, amplitude=amplitude,
							   wavelength=wavelength)
	R = 2 / 3 # density ratio for a neutrally buoyant particle
	x_0 = 0
	Fr = my_wave.froude_num
	T = Fr * my_wave.angular_freq
	delta_t = 5e-3 * T
	num_periods = 20 * T

	# run numerics and compute drift velocity for each Stokes number
	for St in stokes_nums:
		my_particle = prt.Particle(stokes_num=St)
		my_system = ts.MyTransportSystem(my_particle, my_wave, R)
		u_d_list = []

		for z_0 in initial_depths:
			# run numerics
			xdot_0, zdot_0 = my_wave.velocity(x_0, z_0, t=0)
			x, z, xdot, zdot, t = my_system.run_numerics(include_history=False,
											x_0=x_0, z_0=z_0,
											xdot_0=xdot_0, zdot_0=zdot_0,
											delta_t=delta_t,
											num_periods=num_periods)
			u_d, _ = compute_drift_velocity(x, z, xdot, t)
			u_d_list.append(u_d / Fr)
		my_dict['u_d_%g' % St] = u_d_list

	# write results to data file
	my_dict = dict([(key, pd.Series(value)) for key, value in my_dict.items()])
	numerics = pd.DataFrame(my_dict)
	numerics.to_csv('../data/deep_water_wave/drift_velocity_varying_st.csv',
					index=False)

def compute_drift_velocity(x, z, xdot, t):
	r"""
	Computes the Stokes drift velocity
	$$\mathbf{u}_d = \langle u_d, w_d \rangle$$
	using the distance travelled by the particle averaged over each wave period,
	$$\mathbf{u}_d = \frac{\mathbf{x}_{n + 1} - \mathbf{x}_n}{\text{period}}.$$

	Parameters
	----------
	x : array
		The horizontal positions used to evaluate the drift velocity.
	z : array
		The vertical positions used to evaluate the drift velocity.
	xdot : array
		The horizontal velocities used to evaluate the drift velocity.
	t : array
		The times used to evaluate the drift velocity.

	Returns
	-------
	u_d : float
		The horizontal Stokes drift velocity.
	w_d : float
		The vertical Stokes drift velocity.
	"""
	u_d, w_d = [], []

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
	u_d_list, w_d_list = [], []
	for i in range(1, len(interpd_t)):
		u_d_list.append((interpd_x[i] - interpd_x[i - 1])
				 / (interpd_t[i] - interpd_t[i - 1]))
		w_d_list.append((interpd_z[i] - interpd_z[i - 1])
						/ (interpd_t[i] - interpd_t[i - 1]))
	u_d = np.mean(u_d_list)
	w_d = np.mean(w_d_list)
	return u_d, w_d

if __name__ == '__main__':
	main()

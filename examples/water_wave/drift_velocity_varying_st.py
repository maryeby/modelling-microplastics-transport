import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
import pandas as pd
import scipy.constants as constants

from transport_framework import particle as prt
from models import water_wave as fl
from models import my_system as ts

def main():
	r"""
	This program numerically computes the horizontal Stokes drift velocity of an
	inertial particle in linear water waves for various Stokes numbers and saves
	the results to the `data/water_wave` directory.
	"""
	# initialize Stokes numbers, depths, & create a dictionary to store results
	stokes_nums = [0.01, 0.1, 1, 10]
#	stokes_nums = [0.1]
	h = [10, 2, 1] # depths
	my_dict = {}
	my_dict['St'] = stokes_nums

	# run numerics and compute drift velocity for each Stokes number
	for St in stokes_nums:
		my_particle = prt.Particle(stokes_num=St)
		deep_u_d, _ = compute_drift_velocity('deep', h[0], my_dict, my_particle,
											 include_history=False)
		intermediate_u_d, _ = compute_drift_velocity('intermediate', h[1],
													 my_dict, my_particle,
													 include_history=False)
		shallow_u_d, _ = compute_drift_velocity('shallow', h[2], my_dict,
												my_particle,
												include_history=False)
		deep_u_d_history, _ = compute_drift_velocity('deep', h[0], my_dict,
													 my_particle,
													 include_history=True)
		intermediate_u_d_history, _ = compute_drift_velocity('intermediate',
										h[1], my_dict, my_particle,
											include_history=True)
		shallow_u_d_history, _ = compute_drift_velocity('shallow', h[2],
														my_dict, my_particle,
														include_history=True)
		# store results
		my_dict['deep_u_d_%g' % St] = deep_u_d
		my_dict['intermediate_u_d_%g' % St] = intermediate_u_d
		my_dict['shallow_u_d_%g' % St] = shallow_u_d
		my_dict['deep_u_d_history_%g' % St] = deep_u_d_history
		my_dict['intermediate_u_d_history_%g' % St] = intermediate_u_d_history
		my_dict['shallow_u_d_history_%g' % St] = shallow_u_d_history

	# write results to data file
	my_dict = dict([(key, pd.Series(value)) for key, value in my_dict.items()])
	numerics = pd.DataFrame(my_dict)
	numerics.to_csv('../data/water_wave/drift_velocity_varying_st.csv',
					index=False)

def compute_drift_velocity(label, depth, my_dict, particle, include_history):
	r"""
	Computes the Stokes drift velocity
	$$\mathbf{u}_d = \langle u_d, w_d \rangle$$
	using the distance travelled by the particle averaged over each wave period,
	$$\mathbf{u}_d = \frac{\mathbf{x}_{n + 1} - \mathbf{x}_n}{\text{period}}.$$

	Parameters
	----------
	label : str
		The label used to store the initial depth values.
	depth : int
		The depth of the water.
	my_dict : dictionary
		The dictionary to store the initial depths.
	particle : Particle (obj)
		The particle travelling through the wave.
	include_history : boolean
		Whether to include history effects.

	Returns
	-------
	u_d : float
		The normalized horizontal Stokes drift velocity.
	w_d : float
		The normalized vertical Stokes drift velocity.
	"""
	# initialize and store initial depths
	initial_depths = np.linspace(0, -depth, 4, endpoint=False)
	my_dict[label + '_z'] = initial_depths
	my_dict[label + '_h'] = depth
	my_dict[label + '_z/h'] = initial_depths / depth

	# initialize and store wave parameters
	amplitude, wavelength = 0.02, 12
	if label == 'deep':
		my_dict['amplitude'] = amplitude
		my_dict['wavelength'] = wavelength

	# initialize the flow (wave) and transport system
	my_wave = fl.WaterWave(depth=depth, amplitude=amplitude,
						   wavelength=wavelength)
	R = 2 / 3 # neutrally buoyant particle
	my_system = ts.MyTransportSystem(particle, my_wave, R)

	# initialize parameters for the numerics
	x_0 = 0
	T =  my_wave.froude_num * my_wave.angular_freq
	delta_t = 1e-2 * T
	num_periods = 20 * T

	u_d, w_d = [], []
	for z_0 in initial_depths:
		# run numerics
		xdot_0, zdot_0 = my_wave.velocity(x_0, z_0, t=0)
		x, z, xdot, zdot, t = my_system.run_numerics(x_0=x_0, z_0=z_0,
										xdot_0=xdot_0, zdot_0=zdot_0,
										include_history=include_history,
										delta_t=delta_t,
										num_periods=num_periods)
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
		avg_u_d = np.mean(u_d_list)
		avg_w_d = np.mean(w_d_list)
		u_d.append(avg_u_d)
		w_d.append(avg_w_d)

	# normalize and return results
	Fr = my_wave.froude_num
	u_d = np.array(u_d) / Fr
	w_d = np.array(w_d) / Fr
	return u_d, w_d

if __name__ == '__main__':
	main()

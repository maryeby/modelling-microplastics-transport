import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
import pandas as pd
import itertools
import scipy.constants as constants
from parallelbar import progress_starmap

from transport_framework import particle as prt
from models import deep_water_wave as fl
from models import my_system as ts

def main():
	r"""
	This program numerically computes the Stokes drift velocity of an inertial
	particle in linear deep water waves for various Stokes numbers, and saves
	the results to the `data/deep_water_wave` directory.
	"""
	# initialize stokes numbers and wave parameters
#	stokes_nums = [0.01, 0.1, 1, 10]
	stokes_nums = [0.01, 0.1, 1]
	depth = 10
	amplitude = 0.01
	wavelength = 2
	initial_depths = np.linspace(0, -depth, 10, endpoint=False)

	# initialize the wave object and density ratio to use in the numerics
	my_wave = fl.DeepWaterWave(depth=depth, amplitude=amplitude,
							   wavelength=wavelength)
	R = 2 / 3 # density ratio for a neutrally buoyant particle

	# run numerics in parallel for each Stokes number and initial depth
	stokes_nums, initial_depths = zip(*itertools.product(stokes_nums,
														 initial_depths))
	params = zip(stokes_nums, initial_depths, itertools.repeat(my_wave),
				 itertools.repeat(R))
	num_tasks = len(stokes_nums) * len(initial_depths)
	results = progress_starmap(run_numerics, params, n_cpu=8,
										   total=num_tasks)
	# write results to data file
	numerics = pd.DataFrame(results)
	numerics.to_csv('../data/deep_water_wave/drift_velocity_varying_st.csv',
					index=False)

def run_numerics(stokes_num, z_0, my_wave, density_ratio):
	"""
	Runs a numerical simulation for the specified parameters and gets the
	Stokes drift velocity.

	Parameters
	----------
	stokes_num : float
		The Stokes number to use for the initialization of the particle.
	z_0 : float
		The initial position of the particle.
	my_wave : Wave (obj)
		The wave through which the particle is transported.
	density_ratio : float
		The ratio between the particle and fluid densities.
	
	Returns
	-------
	dict
		Dictionary containing the components of the Stokes drift velocity and
		other relevant parameters.
	"""
	# initialize local parameters for the numerical computations
	x_0 = 0
	Fr = my_wave.froude_num
	T = Fr * my_wave.angular_freq
	delta_t = 5e-3 * T
	num_periods = 20 * T

	# initialize the particle and transport system
	my_particle = prt.Particle(stokes_num)
	my_system = ts.MyTransportSystem(my_particle, my_wave, density_ratio)

	# run numerics
	xdot_0, zdot_0 = my_wave.velocity(x_0, z_0, t=0)
	x, z, xdot, zdot, t = my_system.run_numerics(include_history=False,
									x_0=x_0, z_0=z_0,
									xdot_0=xdot_0, zdot_0=zdot_0,
									delta_t=delta_t,
									num_periods=num_periods)
	u_d, w_d, z_star = compute_drift_velocity(x, z, xdot, t)
	u_d /= Fr # scaling results
	w_d /= Fr

	# save and return results
	results = {'x_0': x_0, 'z_0': z_0, 'z_star': z_star, 'u_d': u_d, 'w_d': w_d,
			   'St': stokes_num, 'h': my_wave.depth, 'A': my_wave.amplitude,
			   'wavelength': my_wave.wavelength, 'k': my_wave.wavenum,
			   'omega': my_wave.angular_freq, 'Fr': my_wave.froude_num,
			   'delta_t': delta_t}
	return results

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
	z_star : float
		The vertical position of the 9 o'clock position of the first orbit.
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
	z_star = interpd_z[0]
	return u_d, w_d, z_star

if __name__ == '__main__':
	main()

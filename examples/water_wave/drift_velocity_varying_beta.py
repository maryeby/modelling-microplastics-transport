import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
import pandas as pd

from transport_framework import particle as prt
from models import water_wave as fl
from models import my_system as ts

def main():
	"""
	This program computes numerical solutions for the Stokes drift velocity of
	inertial particles of varying buoyancies in linear water waves and saves the
	results to the `data/water_wave` directory.
	"""
	# initialize the flow (wave)
	my_dict = {}
	depth = 50
	amplitude = 0.02
	wavelength = 1
	my_dict['amplitude'] = amplitude
	my_dict['wavelength'] = wavelength
	my_wave = fl.WaterWave(depth=depth, amplitude=amplitude,
						   wavelength=wavelength)

	# initialize the parameters for the numerics
	Fr = my_wave.froude_num
	T = my_wave.angular_freq * Fr
	num_periods = 20 * T
	delta_t = 1e-2 * T
	x_0, z_0 = 0, 0
	xdot_0, zdot_0 = my_wave.velocity(x_0, z_0, t=0)
	betas = [0.1, 0.5, 0.9, 1]
#	betas = [0.9]
	my_dict['beta'] = betas

	for beta in betas:
		R = 2 / 3 * beta
		my_particle = prt.Particle(stokes_num=0.157 * R * Fr)
		my_system = ts.MyTransportSystem(my_particle, my_wave, R)

		# run numerics without history
		x, z, xdot, _, t = my_system.run_numerics(include_history=False,
												  x_0=x_0, z_0=z_0,
												  xdot_0=xdot_0, zdot_0=zdot_0,
												  delta_t=delta_t,
												  num_periods=num_periods)
		# compute drift velocity & store results
		u_d, w_d, t_d = compute_drift_velocity(x, z, xdot, t)
		my_dict['x_%g' % beta] = x
		my_dict['z_%g' % beta] = z
		my_dict['u_d_%g' % beta] = u_d
		my_dict['w_d_%g' % beta] = w_d
		my_dict['t_%g' % beta] = t_d / Fr

		# run numerics with history
#		x, z, xdot, _, t = my_system.run_numerics(include_history=True,
#												  x_0=x_0, z_0=z_0,
#												  xdot_0=xdot_0, zdot_0=zdot_0,
#												  delta_t=delta_t,
#												  num_periods=num_periods)
		# compute drift velocity & store results
#		u_d, w_d, t_d = compute_drift_velocity(x, z, xdot, t)
#		my_dict['x_history_%g' % beta] = x
#		my_dict['z_history_%g' % beta] = z
#		my_dict['u_d_history_%g' % beta] = u_d / Fr
#		my_dict['w_d_history_%g' % beta] = w_d / Fr
#		my_dict['t_history_%g' % beta] = t_d / Fr

	# write results to data file
	my_dict = dict([(key, pd.Series(value)) for key, value in my_dict.items()])
	numerics = pd.DataFrame(my_dict)
	numerics.to_csv('../data/water_wave/drift_velocity_varying_beta.csv',
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

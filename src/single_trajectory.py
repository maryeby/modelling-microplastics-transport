import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import constants
from maxey_riley import particle_trajectory

# define variables
amplitude = 0.1						# A
wavelength = 10 					# lambda
wave_num = 2 * np.pi / wavelength 	# k
depth = 2							# h
# calculate omega using dispersion relation
angular_freq = np.sqrt(constants.g * wave_num * np.tanh(wave_num * depth))
period = 2 * np.pi / angular_freq	# period of particle oscillation
t_span = (0, 20 * period)			# time span
density = 2 / 3						# R
stokes_num = 1e-5					# St
x_0, z_0, u_0, w_0 = 0, -0.5, 1, 0	# initial position and velocity values

# compute particle trajectory
x, z = particle_trajectory(x_0, z_0, u_0, w_0, period, wave_num, depth,
						   angular_freq, amplitude, density, stokes_num)

# plot results
plt.plot(x, z, 'k')
plt.title(r'Particle Trajectory with $ St $ = {:.0e}'.format(stokes_num))
plt.xlabel('Horizontal x')
plt.ylabel('Depth z')
plt.show()

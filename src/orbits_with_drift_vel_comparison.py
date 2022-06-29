import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import constants
from maxey_riley import particle_trajectory, compare_drift_velocities

# define variables
amplitude = 0.1						# A
wavelength = 10 					# lambda
wave_num = 2 * np.pi / wavelength	# k
depth = 2							# h
# calculate omega using dispersion relation
angular_freq = np.sqrt(constants.g * wave_num * np.tanh(wave_num * depth))
period = 2 * np.pi / angular_freq	# period of particle oscillation
density = 2 / 3						# R
stokes_num = 1e-20					# St
trajectory_depths = np.linspace(-1, -depth, num=10, endpoint=False)
drift_vel_depths = np.linspace(-1, -depth, num=10, endpoint=False)
x_0, u_0, w_0 = 0, 1, 0				# initial position and velocity
plt.figure()	

# compute particle trajectories from various initial depths
for z_0 in trajectory_depths:
	x, z = particle_trajectory(x_0, z_0, u_0, w_0, period, wave_num, depth,
							   angular_freq, amplitude, density, stokes_num)

	# plot particle trajectories
	plt.subplot(121)
	plt.plot(x, z, 'k')
plt.title('Particle Trajectory')
plt.xlabel('Horizontal x')
plt.ylabel('Depth z')

# compute the drift velocity comparisons for various initial depths	
numerical_drift_vels, analytical_drift_vels = compare_drift_velocities(
												drift_vel_depths, x_0, period,
												wave_num, depth, angular_freq,
												amplitude, density, stokes_num)

# plot drift velocity comparisons
plt.subplot(122)
plt.plot(analytical_drift_vels, drift_vel_depths, 'm--',
		 label='Analytical solution')
plt.scatter(numerical_drift_vels, drift_vel_depths, c='k', marker='^',
			label='Numerical solution')
plt.title('Drift Velocity Comparison')
plt.xlabel(r'Drift velocity $ u_d $')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import constants
from tqdm import tqdm

import ocean_wave

# define variables
my_wave = ocean_wave.OceanWave(amplitude=0.3, depth=10, stokes_num=1e-10)
trajectory_depths = np.linspace(-1, -my_wave.get_depth(), num=5,
								endpoint=False)
drift_vel_depths = np.linspace(-1, -my_wave.get_depth(), num=10,
								endpoint=False)
x_0, u_0, w_0 = 0, 1, 0	# initial position and velocity
plt.figure()	

# compute particle trajectories from various initial depths
for z_0 in tqdm(trajectory_depths):
	x, z = my_wave.particle_trajectory(x_0, z_0, u_0, w_0)
	# plot particle trajectories
	plt.subplot(121)
	plt.plot(x, z, 'k')
plt.title('Particle Trajectory')
plt.xlabel('Horizontal x')
plt.ylabel('Depth z')

# compute the drift velocity comparisons for various initial depths	
numerical_drift_vels, analytical_drift_vels = my_wave.compare_drift_velocities(
													  drift_vel_depths, x_0)

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

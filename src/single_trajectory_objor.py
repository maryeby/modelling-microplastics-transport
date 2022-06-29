import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import constants

import ocean_wave

# define variables
my_wave = ocean_wave.OceanWave(stokes_num=1e-5)
x_0, z_0, u_0, w_0 = 0, -0.5, 1, 0	# initial position and velocity values

# compute particle trajectory with and without history
x, z, x_hist, z_hist = my_wave.particle_trajectory(x_0, z_0, u_0, w_0)

# plot results
plt.plot(x, z, 'k', label='Without history')
plt.plot(x_hist, z_hist, 'k--', label='With history')
plt.title(r'Particle Trajectory with $ St $ = {:.0e}'.format(
													  my_wave.get_stokes_num()))
plt.xlabel('Horizontal x')
plt.ylabel('Depth z')
plt.legend()
plt.show()

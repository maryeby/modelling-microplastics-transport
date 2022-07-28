import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import constants

import ocean_wave

# define variables
my_wave = ocean_wave.OceanWave()
x_0, z_0 = 0, -3
u_0, w_0 = 0, 0
x, z = my_wave.particle_trajectory(my_wave.mr_no_history, x_0, z_0, u_0, w_0)

# plot results
plt.plot(x, z, 'k')
plt.title('Trajectory for a Neutrally Buoyant Particle', fontsize=16)
plt.xlabel('Horizontal x', fontsize=12)
plt.ylabel('Depth z', fontsize=12)
plt.show()

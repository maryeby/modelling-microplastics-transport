import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import constants

import ocean_wave

# define variables
my_wave = ocean_wave.OceanWave()
x_0, z_0 = 0, 0
u_0, w_0 = my_wave.fluid_velocity(x_0, z_0, 0)
x, z = my_wave.particle_trajectory(my_wave.mr_no_history, x_0, z_0, u_0, w_0)

# plot results
plt.plot(my_wave.get_wave_num() * x, my_wave.get_wave_num() * z, 'k')
plt.xlim(0, 3)
plt.ylim(-4, 0)
plt.title(r'Particle Trajectory with $ R $ = {:.2f}'.format(
													 my_wave.get_density()))
plt.xlabel('x')
plt.ylabel('z')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import constants

import ocean_wave

# define variables
my_wave = ocean_wave.OceanWave(amplitude=0.026, wavelength=0.5,
							   density=1.01 * (2 / 3),
							   stokes_num=(0.5 * 1.01 * (2 / 3) / 11))
x_0, z_0, u_0, w_0 = 0.13, -0.3, 0, 0
x, z = my_wave.particle_trajectory(x_0, z_0, u_0, w_0)

# plot results
plt.plot(my_wave.get_wave_num() * x, my_wave.get_wave_num() * z, 'k')
plt.xlim(0, 3)
plt.ylim(-4, 0)
plt.title(r'Particle Trajectory with $ St $ = {:.0e}'.format(
													  my_wave.get_stokes_num()))
plt.xlabel('kx')
plt.ylabel('kz')
plt.show()

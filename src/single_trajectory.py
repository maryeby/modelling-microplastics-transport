import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import constants

import ocean_wave

# define variables
my_wave = ocean_wave.OceanWave(amplitude=0.026, wavelength=0.5, depth=15,
							   beta = 0.99)
x_0, z_0 = 0, 0
u_0, w_0 = 0, 0
x, z = my_wave.particle_trajectory(my_wave.santamaria, x_0, z_0, u_0, w_0)

# plot results
plt.plot(my_wave.get_wave_num() * x, my_wave.get_wave_num() * z, 'k')
plt.title(r'Particle Trajectory for $ \beta = 0.99 $', fontsize=16)
plt.xlabel('kx', fontsize=12)
plt.ylabel('kz', fontsize=12)
plt.xlim(0, 3)
plt.ylim(-4, 0)
plt.show()

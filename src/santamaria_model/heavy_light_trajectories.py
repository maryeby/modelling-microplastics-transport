import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate 

import santamaria

# define waves
heavy_wave = santamaria.SantamariaModel(wave_num=4 * np.pi, amplitude=0.026,
							 			stokes_num=0.5, beta=0.99)
light_wave = santamaria.SantamariaModel(wave_num=4 * np.pi, amplitude=0.026,
							 			stokes_num=0.5, beta=1.01)
# compute trajectories
x_heavy, z_heavy, _, _, _ = heavy_wave.particle_trajectory(0, 0)
x_light, z_light, _, _, _ = light_wave.particle_trajectory(0.13, -0.3)

# plot results
plt.title('Santamaria Figure 1 Reproduction', fontsize=16)
plt.xlabel('kx', fontsize=12)
plt.ylabel('kz', fontsize=12)
plt.xlim(0, 3)
plt.ylim(-3.78, 0)
plt.plot(heavy_wave.get_wave_num() * x_heavy,
		 heavy_wave.get_wave_num() * z_heavy, 'k', label='heavy particle')
plt.plot(light_wave.get_wave_num() * x_light,
		 light_wave.get_wave_num() * z_light, c='coral', label='light particle')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import ocean_wave

# define variables
my_wave = ocean_wave.OceanWave()
x_0, z_0, u_0, w_0 = 0, 0, 0, 0
u, w, time = my_wave.particle_velocity(my_wave.mr_no_history, x_0, z_0, u_0,
									   w_0)
u_hist, w_hist, time_hist = my_wave.particle_velocity(my_wave.mr_with_history,
													  x_0, z_0, u_0, w_0)
velocity = np.linalg.norm(np.array([u, w]), axis=0)
velocity_hist = np.linalg.norm(np.array([u_hist, w_hist]), axis=0)

# plot results
plt.plot(velocity, time, 'k-', label='Without history')
plt.plot(velocity_hist, time_hist, 'k--', label='With history')
# plt.xlim(0, 15)
# plt.ylim(0, 100)
# plt.yscale('log')
plt.xlabel('x')
plt.ylabel('z')
plt.legend()
plt.show()

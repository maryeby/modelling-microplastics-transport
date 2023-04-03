import numpy as np
import matplotlib.pyplot as plt

import transport_system
from models import haller, modified_santamaria

my_system = transport_system.TransportSystem(amplitude=0.02, wavelength=1,
											 density=2/3 * 0.9,
											 stokes_num=0.01)
# generate results from each model
x, z, xdot, zdot, t = my_system.particle_trajectory(haller)
haller = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}
x, z, xdot, zdot, t = my_system.particle_trajectory(modified_santamaria)
new_sm = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}

k = my_system.get_wavenum()

plt.figure(1)
plt.title('Particle Trajectory Comparison', fontsize=14)
plt.xlabel('kx', fontsize=12)
plt.ylabel('kz', fontsize=12)
plt.plot(k * haller['x'], k * haller['z'], c='k', label='Haller (3)')
plt.plot(k * new_sm['x'], k * new_sm['z'], c='hotpink',
		 label=r'My calculated (6*) and (7*)')
plt.legend()
plt.show()

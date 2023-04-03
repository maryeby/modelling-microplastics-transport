import numpy as np
import matplotlib.pyplot as plt

import transport_system
from models import santamaria, santamaria_order0, santamaria_order1, \
				   santamaria_order2

my_system = transport_system.TransportSystem(amplitude=0.02, wavelength=1,
											 density=2/3 * 0.9,
											 stokes_num=0.01, beta=0.9)
# generate results from each model
x, z, xdot, zdot, t = my_system.particle_trajectory(santamaria,
													dimensional=True)
santamaria = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}
x, z, xdot, zdot, t = my_system.particle_trajectory(santamaria_order0,
													dimensional=True)
santamaria_order0 = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}
x, z, xdot, zdot, t = my_system.particle_trajectory(santamaria_order1,
													dimensional=True)
santamaria_order1 = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}
x, z, xdot, zdot, t = my_system.particle_trajectory(santamaria_order2,
													dimensional=True)
santamaria_order2 = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}

omega = my_system.get_angular_freq()
k = my_system.get_wavenum()

plt.figure(1)
plt.title('Particle Trajectory for the Santamaria Model', fontsize=14)
plt.xlabel('kx', fontsize=12)
plt.ylabel('kz', fontsize=12)
plt.plot(k * santamaria['x'], k * santamaria['z'], c='k',
		 label='Santamaria (3) and (4)')
plt.plot(k * santamaria_order0['x'], k * santamaria_order0['z'], c='hotpink',
		 label=r'Santamaria (5) order $ \tau^0 $')
plt.plot(k * santamaria_order1['x'], k * santamaria_order1['z'],
		 c='mediumpurple', label=r'Santamaria (5) order $ \tau^1 $')
plt.plot(k * santamaria_order2['x'], k * santamaria_order2['z'],
		 c='cornflowerblue', label=r'Santamaria (5) order $ \tau^2 $')
plt.legend()
plt.show()

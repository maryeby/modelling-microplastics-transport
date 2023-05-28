import numpy as np
import matplotlib.pyplot as plt

import transport_system
from models import santamaria, santamaria_order0, santamaria_order1, \
				   santamaria_order2

my_system = transport_system.TransportSystem(amplitude=0.02, wavelength=1,
											 stokes_num=0.1, beta=0.99)
# generate results from each model
x, z, xdot, zdot, t = my_system.particle_trajectory(santamaria, deep=True,
													dimensional=True)
santamaria = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}
#x, z, xdot, zdot, t = my_system.particle_trajectory(santamaria_order0,
#													deep=True, dimensional=True)
#santamaria_order0 = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}
x, z, xdot, zdot, t = my_system.particle_trajectory(santamaria_order1,
													deep=True, dimensional=True)
santamaria_order1 = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}
x, z, xdot, zdot, t = my_system.particle_trajectory(santamaria_order2,
													deep=True, dimensional=True)
santamaria_order2 = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}

k = my_system.get_wavenum()
beta = my_system.get_beta()
st = my_system.get_stokes_num()

plt.figure(1)
plt.title('First Model Particle Trajectory with '
		  + r'$\beta = {:.2f}, St = {:.2f}$'.format(beta, st), fontsize=14)
plt.xlabel('kx', fontsize=12)
plt.ylabel('kz', fontsize=12)
plt.plot(k * santamaria['x'], k * santamaria['z'], c='k', zorder=2,
		 label='Numerical results')
#plt.plot(k * santamaria_order0['x'], k * santamaria_order0['z'], c='hotpink',
#		 label='Leading order')
plt.plot(k * santamaria_order1['x'], k * santamaria_order1['z'],
		 c='hotpink', label='First order')
plt.plot(k * santamaria_order2['x'], k * santamaria_order2['z'],
		 c='cornflowerblue', label='Second order')
plt.legend()
plt.show()

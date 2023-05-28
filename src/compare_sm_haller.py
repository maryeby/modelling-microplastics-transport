import numpy as np
import matplotlib.pyplot as plt

import transport_system
from models import haller, haller_order0, haller_order1, haller_order2, \
				   santamaria_order2

my_system = transport_system.TransportSystem(amplitude=0.02, wavelength=1,
											 stokes_num=0.1, beta=0.99)
# generate results from each model
x, z, xdot, zdot, t = my_system.particle_trajectory(haller, deep=True)
numerics = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}
x, z, xdot, zdot, t = my_system.particle_trajectory(haller_order0, deep=True)
haller_order0 = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}
x, z, xdot, zdot, t = my_system.particle_trajectory(haller_order1, deep=True)
haller_order1 = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}
x, z, xdot, zdot, t = my_system.particle_trajectory(haller_order2, deep=True)
haller_order2 = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}
x, z, xdot, zdot, t = my_system.particle_trajectory(santamaria_order2,
													dimensional=True, deep=True)
santamaria = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}

# retrieve relevant variables needed for plotting
k = my_system.get_wavenum()
beta = my_system.get_beta()
density = my_system.get_density()
st = my_system.get_stokes_num()

# plot results
plt.figure(dpi=200)
plt.title('Numerical and Inertial Equation Results with '
		  + r'$R = {:.2f}, St = {:.2f}$'.format(density, st),
		  fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('z', fontsize=12)
plt.plot('x', 'z', c='k', data=numerics, zorder=5, label='Numerical')
plt.plot('x', 'z', c='grey', data=haller_order0, zorder=4,
		 label='Leading order')
plt.plot('x', 'z', c='mediumpurple', data=haller_order1, zorder=3,
		 label='First order')
plt.plot('x', 'z', c='cornflowerblue', data=haller_order2, zorder=2,
		 label='Second order (Haller)')
plt.plot(k * santamaria['x'], k * santamaria['z'], c='hotpink', zorder=1,
		 label='Second order (Santamaria)')
plt.legend()
plt.show()

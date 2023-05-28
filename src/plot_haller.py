import numpy as np
import matplotlib.pyplot as plt

import transport_system
from models import haller, haller_order0, haller_order1, haller_order2, \
				   haller_order2_no_jacobian

my_system = transport_system.TransportSystem(amplitude=0.02, wavelength=1,
											 beta=0.99, stokes_num=0.1)
# generate results from each model
x, z, xdot, zdot, t = my_system.particle_trajectory(haller)
original = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}
#x, z, xdot, zdot, t = my_system.particle_trajectory(haller_order0)
#order0 = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}
x, z, xdot, zdot, t = my_system.particle_trajectory(haller_order1)
order1 = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}
x, z, xdot, zdot, t = my_system.particle_trajectory(haller_order2)
order2_with_jacobian = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}
#x, z, xdot, zdot, t = my_system.particle_trajectory(haller_order2_no_jacobian)
#order2_no_jacobian = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}

# retrieve relevant variables needed for plotting
beta = my_system.get_beta()
st = my_system.get_stokes_num()

# plot results
plt.figure()
plt.title('Second Model Particle Trajectories with '
		  + r'$\beta = {:.2f},St = {:.2f}$'.format(beta, st), fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('z', fontsize=12)
plt.plot('x', 'z', c='k', data=original, label='Numerical results')
#plt.plot('x', 'z', c='hotpink', data=order0, label='Leading order')
plt.plot('x', 'z', c='hotpink', data=order1, label='First order')
plt.plot('x', 'z', c='cornflowerblue', data=order2_with_jacobian,
		 label='Second order')
#plt.plot('x', 'z', c='mediumseagreen', data=order2_no_jacobian,
#		 label='Second order without Jacobian contribution')
plt.legend()
plt.show()

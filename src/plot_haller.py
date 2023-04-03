import numpy as np
import matplotlib.pyplot as plt

import transport_system
from models import haller, haller_order0, haller_order1, haller_order2, \
				   haller_order2_no_jacobian

my_system = transport_system.TransportSystem(amplitude=0.02, wavelength=1,
											 density=2/3 * 0.9,
											 stokes_num=0.01)
# generate results from each model
x, z, xdot, zdot, t = my_system.particle_trajectory(haller)
original = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}
x, z, xdot, zdot, t = my_system.particle_trajectory(haller_order0)
order0 = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}
x, z, xdot, zdot, t = my_system.particle_trajectory(haller_order1)
order1 = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}
x, z, xdot, zdot, t = my_system.particle_trajectory(haller_order2)
order2_with_jacobian = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}
x, z, xdot, zdot, t = my_system.particle_trajectory(haller_order2_no_jacobian)
order2_no_jacobian = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}

k = my_system.get_wavenum()

plt.figure(1)
plt.title('Particle Trajectories for the Haller Model', fontsize=14)
plt.xlabel('kx', fontsize=12)
plt.ylabel('kz', fontsize=12)
plt.plot(k * original['x'], k * original['z'], c='k', label='Haller (3)')
plt.plot(k * order0['x'], k * order0['z'], c='hotpink',
		 label=r'Haller (10) order $ \epsilon^0 $')
plt.plot(k * order1['x'], k * order1['z'], c='mediumpurple',
		 label=r'Haller (10) order $ \epsilon^1 $')
plt.plot(k * order2_with_jacobian['x'],
		 k * order2_with_jacobian['z'], c='cornflowerblue',
		 label=r'Haller (10) order $ \epsilon^2 $ with Jacobian contribution')
plt.plot(k * order2_no_jacobian['x'],
		 k * order2_no_jacobian['z'], c='lime',
		 label=r'Haller (10) order $ \epsilon^2 $ without Jacobian contribution')
plt.legend()
plt.show()

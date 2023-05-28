import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import transport_system
from models import santamaria

heavy_beta = 0.96
light_beta = 1.04
cathals_data = pd.read_csv('cathals_data.csv')

heavy_system = transport_system.TransportSystem(amplitude=0.026,
												  wavelength=0.5,
												  beta=heavy_beta,
												  stokes_num=0.5)
light_system = transport_system.TransportSystem(amplitude=0.026,
												  wavelength=0.5,
												  beta=light_beta,
												  stokes_num=0.5)
k = heavy_system.get_wavenum()
# generate results from each model
x, z, xdot, zdot, t = heavy_system.particle_trajectory(santamaria, deep=True,
													   dimensional=True,
													   num_periods=100)
heavy = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}
x, z, xdot, zdot, t = light_system.particle_trajectory(santamaria, deep=True,
													   dimensional=True,
													   x_0=0.13, z_0=-0.4,
													   num_periods=100)
light = {'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot, 't': t}
print(heavy_system.get_density())
print(light_system.get_density())

plt.figure()
plt.xlabel('kx', fontsize=16)
plt.ylabel('kz', fontsize=16)
plt.axis([0, 3.2, -4, 0])
plt.xticks(fontsize=14)
plt.yticks([-3, -2, -1, 0], fontsize=14)
#plt.plot('heavy_x', 'heavy_z', c='cornflowerblue', lw=2, marker='.',
#		 data=cathals_data, label=r'$\beta = ${:.2f}'.format(heavy_beta))
#plt.plot('light_x', 'light_z', c='coral', lw=2, marker='.',
#		 data=cathals_data, label=r'$\beta = ${:.2f}'.format(light_beta))
plt.plot(k * heavy['x'], k * heavy['z'], c='k')#,
#		 label='R = {:.2f}'.format(heavy_beta))
plt.plot(k * light['x'], k * light['z'], 'k:')#,
#		 label='R = {:.2f}'.format(light_beta))
#plt.plot('x', 'z', c='k', data=heavy,
#		 label=r'$\beta = ${:.2f}'.format(heavy_beta))
#plt.plot('x', 'z', 'k:', data=light,
#		 label=r'$\beta = ${:.2f}'.format(light_beta))
#plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate 

import santamaria

my_wave = santamaria.SantamariaModel(wave_num=2 * np.pi, amplitude=0.02,
							 		 stokes_num=0.157, beta=0.9)
# defining local parameters for ease of use
U = my_wave.get_max_velocity()
omega = my_wave.get_angular_freq()
k = my_wave.get_wave_num()

# compute drift velocities
x, z, u, w, t = my_wave.particle_trajectory(0, 0)
previous_t, current_t, previous_u, current_u, previous_w, current_w, \
	interpd_t, interpd_u, interpd_w, \
	interpd_x, interpd_z = my_wave.period_info(x, z, u, w, t)
numerical_u, numerical_w, numerical_t, analytical_ud, analytical_wd, \
	analytical_u, analytical_w = my_wave.drift_velocities(x, z, u, w, t)

# plot results
plt.figure(1)
plt.subplot(211)
plt.title('Horizontal Lagrangian Velocity with Period Endpoints', fontsize=14)
plt.xlabel(r'$ \omega t $', fontsize=12)
plt.ylabel(r'$ u / U $', fontsize=12)
plt.axhline(0, c='silver', linestyle=':')
plt.plot(omega * t, u / U, 'k', zorder=2, label='scaled horizontal velocity')
plt.plot(omega * t, analytical_u / U, c='coral', zorder=1,
		 label='scaled analytical u')
plt.scatter(omega * np.array(current_t), np.array(current_u) / U, c='k',
			zorder=4, label='chosen period endpoint')
plt.scatter(omega * np.array(previous_t), np.array(previous_u) / U, c='coral',
			zorder=3, label='last point before period endpoint')
plt.scatter(omega * np.array(interpd_t), np.array(interpd_u) / U,
			c='mediumpurple', zorder=5, label='interpolated period endpoint')
plt.legend()

plt.subplot(212)
plt.title('Vertical Lagrangian Velocity with Period Endpoints', fontsize=14)
plt.xlabel(r'$ \omega t $', fontsize=12)
plt.ylabel(r'$ w / U $', fontsize=12)
plt.axhline(my_wave.get_settling_velocity() / U, c='k', zorder=1, linestyle=':')
plt.plot(omega * t, w / U, 'k', zorder=3, label='scaled vertical velocity')
plt.plot(omega * t, analytical_w / U, c='coral', zorder=2,
		 label='scaled analytical w')
plt.scatter(omega * np.array(current_t), np.array(current_w) / U, c='k',
			zorder=5, label='chosen period endpoint')
plt.scatter(omega * np.array(previous_t), np.array(previous_w) / U, c='coral',
			zorder=4, label='last point before period endpoint')
plt.scatter(omega * np.array(interpd_t), np.array(interpd_w) / U,
			c='mediumpurple', zorder=6, label='interpolated period endpoint')
plt.legend()

#plt.figure(2)
plt.suptitle('Drift Velocity Comparison', fontsize=16)
plt.subplot(121)
plt.xlabel(r'$ \omega t $', fontsize=12)
plt.ylabel(r'$ u_d / U $', fontsize=12)
plt.xlim(0, 80)
plt.ylim(0, 0.15)
plt.xticks(ticks=range(0, 80, 10))
plt.yticks(ticks=[0, 0.05, 0.1])
plt.plot(my_wave.get_angular_freq() * t, analytical_ud / U, c='k',
		 label='analytical')
plt.scatter(omega * np.array(numerical_t), numerical_u / U, facecolors='none', 
			edgecolors='k', label='numerical')

plt.subplot(122)
plt.xlabel(r'$ \omega t $', fontsize=12)
plt.ylabel(r'$ w_d / U $', fontsize=12)
plt.xlim(0, 80)
plt.ylim(-0.128, -0.1245)
plt.xticks(ticks=range(0, 80, 10))
plt.yticks(ticks=[-0.128, -0.127, -0.126, -0.125])
plt.axhline(my_wave.get_settling_velocity() / U, c='k', linestyle=':')
plt.plot(omega * t, analytical_wd / U, c='k', label='analytical')
plt.scatter(omega * np.array(numerical_t), np.array(numerical_w) / U,
			facecolors='none', edgecolors='k', label='numerical')

plt.figure(3)
plt.title('Particle Trajectory', fontsize=16)
plt.xlabel('kx', fontsize='12')
plt.ylabel('kz', fontsize='12')
plt.plot(k * x, k * z, c='k', marker='.', zorder=3,
		 label='generated data points')
plt.scatter(k * np.array(interpd_x), k * np.array(interpd_z), c='mediumpurple',
			marker='o', zorder=4, label='period endpoints')
plt.legend()
plt.show()

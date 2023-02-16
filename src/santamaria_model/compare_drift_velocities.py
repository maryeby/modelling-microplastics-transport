import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate 

import santamaria

def main():
	delta_t = 5e-5
	is_numerical=True
	my_wave = santamaria.SantamariaModel(wave_num=2 * np.pi, amplitude=0.02,
										 stokes_num=0.157, beta=0.9)
	x, z, u, w, t = my_wave.particle_trajectory(delta_t=delta_t)

	plt.figure(1)
	plot_uw(my_wave, x, z, u, w, t, is_numerical)
	plt.figure(2)
	plot_particle_trajectory(my_wave, x, z, u, w, t)
	plt.figure(3)
	plot_drift_velocity(my_wave, x, z, u, w, t, delta_t, limit_axes=False)
#	plot_drift_velocity_varying_delta_t(my_wave, x, z, u, w, t,
#										limit_axes=False)
	plt.show()

def plot_uw(my_wave, x, z, u, w, t, is_numerical):
	U = my_wave.get_max_velocity()
	omega = my_wave.get_angular_freq()
	analytical_u, analytical_w = my_wave.analytical_particle_velocity(t=t)

	plt.subplot(211)
	plt.title('Horizontal Lagrangian Velocity with Period Endpoints',
			  fontsize=14)
	plt.xlabel(r'$ \omega t $', fontsize=12)
	plt.ylabel(r'$ u / U $', fontsize=12)
	plt.axhline(0, c='silver', zorder=1, linestyle=':')
	plt.plot(omega * t, u / U, 'k', zorder=3,
			 label='scaled horizontal velocity')
	plt.plot(omega * t, analytical_u / U, c='coral', zorder=2,
			 label='scaled analytical u')

	plt.subplot(212)
	plt.title('Vertical Lagrangian Velocity with Period Endpoints', fontsize=14)
	plt.xlabel(r'$ \omega t $', fontsize=12)
	plt.ylabel(r'$ w / U $', fontsize=12)
	plt.axhline(my_wave.get_settling_velocity() / U, c='k', zorder=1,
				linestyle=':')
	plt.plot(omega * t, w / U, 'k', zorder=3, label='scaled vertical velocity')
	plt.plot(omega * t, analytical_w / U, c='coral', zorder=2,
			 label='scaled analytical w')

	if is_numerical:
		previous_t, current_t, previous_u, current_u, previous_w, current_w, \
			interpd_t, interpd_u, interpd_w, interpd_x, \
			interpd_z = my_wave.interpolated_period_info(x, z, u, w, t)

		plt.subplot(211)
		plt.scatter(omega * np.array(current_t), np.array(current_u) / U, c='k',
					zorder=5, label='chosen period endpoint')
		plt.scatter(omega * np.array(previous_t), np.array(previous_u) / U,
					c='coral', zorder=4,
					label='last point before period endpoint')
		plt.scatter(omega * np.array(interpd_t), np.array(interpd_u) / U,
					c='mediumpurple', zorder=6,
					label='interpolated period endpoint')
		plt.legend()

		plt.subplot(212)
		plt.scatter(omega * np.array(current_t), np.array(current_w) / U, c='k',
					zorder=5, label='chosen period endpoint')
		plt.scatter(omega * np.array(previous_t), np.array(previous_w) / U,
					c='coral', zorder=4,
					label='last point before period endpoint')
		plt.scatter(omega * np.array(interpd_t), np.array(interpd_w) / U,
					c='mediumpurple', zorder=6,
					label='interpolated period endpoint')
		plt.legend()
	else:
		_, _, _, current_u, current_w, \
			current_t = my_wave.analytical_period_info(x, z, t)
		plt.subplot(211)
		plt.scatter(omega * np.array(current_t), np.array(current_u) / U, c='k',
					zorder=4, label='chosen period endpoint')
		plt.legend()

		plt.subplot(212)
		plt.scatter(omega * np.array(current_t), np.array(current_w) / U, c='k',
					zorder=4, label='chosen period endpoint')
		plt.legend()

def plot_particle_trajectory(my_wave, x, z, u, w, t):
	k = my_wave.get_wave_num()
	_, _, _, _, _, _, _, _, _, \
	numerical_x, numerical_z = my_wave.interpolated_period_info(x, z, u,
																w, t)
	_, analytical_x, analytical_z, \
	_, _, _ = my_wave.analytical_period_info(x, z, t)

	plt.title('Particle Trajectory', fontsize=16)
	plt.xlabel('kx', fontsize='12')
	plt.ylabel('kz', fontsize='12')
	plt.plot(k * x, k * z, c='k', marker='.', zorder=1,
			 label='generated data points')
	plt.scatter(k * np.array(numerical_x), k * np.array(numerical_z),
				c='mediumpurple', zorder=3,
				label='numerical period endpoints')
	plt.scatter(k * np.array(analytical_x), k * np.array(analytical_z),
				c='coral', zorder=2,
				label='analytical period endpoints')
	plt.legend()

def plot_drift_velocity(my_wave, x, z, u, w, t, delta_t, limit_axes):
	U = my_wave.get_max_velocity()
	omega = my_wave.get_angular_freq()
	analytical_ud, analytical_wd = my_wave.analytical_drift_velocity(t=t)
	my_ud, my_wd = my_wave.my_analytical_drift_velocity(z=z, u=u, w=w, t=t)
	u_d, w_d, numerical_t = my_wave.numerical_drift_velocity(x, z, u, w, t)
	u_avg, w_avg, analytical_t = my_wave.analytical_averages(x, z, u, w, t)

	plt.suptitle('Drift Velocity Comparison with '
				 + r'$\Delta t =$ {:.0e}'.format(delta_t), fontsize=16)
	plt.subplot(121)
	plt.xlabel(r'$ \omega t $', fontsize=12)
	plt.ylabel(r'$ u_d / U $', fontsize=12)
	plt.plot(omega * t, analytical_ud / U, c='k', zorder=1, 
			 label='eq (13) as listed in Santamaria 2013')
	plt.plot(omega * t, my_ud / U, c='mediumpurple', zorder=1,
			 label='eq (13) from our calculations')
	plt.scatter(omega * numerical_t, u_d / U, zorder=3, 
				facecolors='none', edgecolors='k', label='numerical')
#	plt.scatter(omega * analytical_t, u_avg / U, zorder=2, c='k', marker='x',
#				label='average of eq (11)')
	plt.legend()

	plt.subplot(122)
	plt.xlabel(r'$ \omega t $', fontsize=12)
	plt.ylabel(r'$ w_d / U $', fontsize=12)
	plt.axhline(my_wave.get_settling_velocity() / U, c='k', linestyle=':')
	plt.plot(omega * t, analytical_wd / U, c='k', zorder=1,
			 label='eq (14) as listed in Santamaria 2013')
	plt.plot(omega * t, my_wd / U, c='mediumpurple', zorder=1,
			 label='eq (14) from our calculations')
	plt.scatter(omega * numerical_t, w_d / U, zorder=3, 
				facecolors='none', edgecolors='k', label='numerical')
#	plt.scatter(omega * analytical_t, w_avg / U, zorder=2, c='k', marker='x',
#				label='average of eq (12)')
	plt.legend()

	if limit_axes:
		plt.subplot(121)
		plt.xlim(0, 80)
		plt.ylim(0, 0.15)
		plt.xticks(ticks=range(0, 80, 10))
		plt.yticks(ticks=[0, 0.05, 0.1])
		
		plt.subplot(122)
		plt.xlim(0, 80)
		plt.ylim(-0.128, -0.1245)
		plt.xticks(ticks=range(0, 80, 10))
		plt.yticks(ticks=[-0.128, -0.127, -0.126, -0.125])

def plot_drift_velocity_varying_delta_t(my_wave, x, z, u, w, t, limit_axes):
	U = my_wave.get_max_velocity()
	omega = my_wave.get_angular_freq()

	# coarse delta t
	x_coarse, z_coarse, u_coarse, w_coarse, \
		t_coarse = my_wave.particle_trajectory(delta_t=5e-3)
	numerical_u_coarse, numerical_w_coarse, \
		numerical_t_coarse = my_wave.numerical_drift_velocity(x_coarse, 
								 z_coarse, u_coarse, w_coarse, t_coarse)
	# medium delta t
	analytical_ud, analytical_wd = my_wave.analytical_drift_velocity(t=t)
	numerical_u_med, numerical_w_med, \
		numerical_t_med = my_wave.numerical_drift_velocity(x, z, u, w, t)

	# fine delta t
	x_fine, z_fine, u_fine, w_fine, \
		t_fine = my_wave.particle_trajectory(delta_t=5e-7)
	numerical_u_fine, numerical_w_fine, \
		numerical_t_fine = my_wave.numerical_drift_velocity(x_fine,
							   z_fine, u_fine, w_fine, t_fine)
	
	plt.suptitle(r'Drift Velocity Comparison with Varying $\Delta t$',
				 fontsize=16)
	plt.subplot(121)
	plt.xlabel(r'$ \omega t $', fontsize=12)
	plt.ylabel(r'$ u_d / U $', fontsize=12)
	plt.plot(omega * t, analytical_ud / U, c='k', zorder=1, label='analytical')
	plt.scatter(omega * numerical_t_coarse, numerical_u_coarse / U,
				zorder=2, marker='s', facecolors='none', edgecolors='k',
				label=r'coarse $\Delta t$')
	plt.scatter(omega * numerical_t_med, numerical_u_med / U, zorder=2, 
				facecolors='none', edgecolors='k', label=r'medium $\Delta t$')
	plt.scatter(omega * numerical_t_fine, numerical_u_fine / U, zorder=4,
				c='k', marker='x', label=r'fine $\Delta t$')
	plt.legend()

	plt.subplot(122)
	plt.xlabel(r'$ \omega t $', fontsize=12)
	plt.ylabel(r'$ w_d / U $', fontsize=12)
	plt.axhline(my_wave.get_settling_velocity() / U, c='k', linestyle=':')
	plt.plot(omega * t, analytical_wd / U, c='k', zorder=1, label='analytical')
	plt.scatter(omega * numerical_t_coarse,
				numerical_w_coarse / U, zorder=2,
				marker='s', facecolors='none', edgecolors='k',
				label=r'coarse $\Delta t$')
	plt.scatter(omega * numerical_t_med, numerical_w_med / U, zorder=3,
				facecolors='none', edgecolors='k', label=r'medium $\Delta t$')
	plt.scatter(omega * numerical_t_fine, numerical_w_fine / U,
				zorder=4, c='k', marker='x', label=r'fine $\Delta t$')
	plt.legend()

	if limit_axes:
		plt.subplot(121)
		plt.xlim(0, 80)
		plt.ylim(0, 0.15)
		plt.xticks(ticks=range(0, 80, 10))
		plt.yticks(ticks=[0, 0.05, 0.1])
		
		plt.subplot(122)
		plt.xlim(0, 80)
		plt.ylim(-0.128, -0.1245)
		plt.xticks(ticks=range(0, 80, 10))
		plt.yticks(ticks=[-0.128, -0.127, -0.126, -0.125])

main()

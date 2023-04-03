import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

import transport_system
from models import santamaria

def main():
	delta_t = 5e-5
	is_numerical=False
	my_system = transport_system.TransportSystem(wavelength=1, amplitude=0.02,
											   	 stokes_num=0.157, beta=0.9)
	x, z, xdot, zdot, t = my_system.particle_trajectory(santamaria,
														delta_t=delta_t)
	plt.figure(1)
	plot_xdot_zdot(my_system, x, z, xdot, zdot, t, is_numerical)
	plt.figure(2)
	plot_particle_trajectory(my_system, x, z, xdot, zdot, t)
	plt.figure(3)
	plot_drift_velocity(my_system, x, z, xdot, zdot, t, delta_t,
						limit_axes=False)
#	plot_drift_velocity_varying_delta_t(my_system, x, z, xdot, zdot, t,
#										limit_axes=False)
	plt.show()

def plot_xdot_zdot(my_system, x, z, xdot, zdot, t, is_numerical):
	U = my_system.get_max_velocity()
	omega = my_system.get_angular_freq()
	analytical_xdot, \
		analytical_zdot = my_system.analytical_particle_velocity(t=t)

	plt.subplot(211)
	plt.title('Horizontal Lagrangian Velocity with Period Endpoints',
			  fontsize=14)
	plt.xlabel(r'$ \omega t $', fontsize=12)
	plt.ylabel(r'$ \dot{x} / U $', fontsize=12)
	plt.axhline(0, c='silver', zorder=1, linestyle=':')
	plt.plot(omega * t, xdot / U, 'k', zorder=3,
			 label='scaled numerical $ \dot{x} $')
	plt.plot(omega * t, analytical_xdot / U, c='coral', zorder=2,
			 label=r'scaled analytical $ \dot{x} $')

	plt.subplot(212)
	plt.title('Vertical Lagrangian Velocity with Period Endpoints', fontsize=14)
	plt.xlabel(r'$ \omega t $', fontsize=12)
	plt.ylabel(r'$ \dot{z} / U $', fontsize=12)
	plt.axhline(my_system.get_settling_velocity() / U, c='k', zorder=1,
				linestyle=':')
	plt.plot(omega * t, zdot / U, 'k', zorder=3,
			 label='scaled numerical $ \dot{z} $')
	plt.plot(omega * t, analytical_zdot / U, c='coral', zorder=2,
			 label=r'scaled analytical $ \dot{z} $')

	if is_numerical:
		previous_t, current_t, previous_xdot, current_xdot, previous_zdot, \
			current_zdot, interpd_x, interpd_z, interpd_t, interpd_xdot, \
			interpd_zdot = my_system.numerical_period_info(x, z,
															   xdot, zdot, t)

		plt.subplot(211)
		plt.scatter(omega * np.array(current_t), np.array(current_xdot) / U,
					c='mediumpurple', zorder=5,
					label='first point of new period')
		plt.scatter(omega * np.array(previous_t), np.array(previous_xdot) / U,
					c='coral', zorder=4,
					label='last point before period endpoint')
		plt.scatter(omega * np.array(interpd_t), np.array(interpd_xdot) / U,
					c='k', zorder=6, label='interpolated period endpoint')
		plt.legend()

		plt.subplot(212)
		plt.scatter(omega * np.array(current_t), np.array(current_zdot) / U,
					c='mediumpurple', zorder=5,
					label='first point of new period')
		plt.scatter(omega * np.array(previous_t), np.array(previous_zdot) / U,
					c='coral', zorder=4,
					label='last point before period endpoint')
		plt.scatter(omega * np.array(interpd_t), np.array(interpd_zdot) / U,
					c='k', zorder=6, label='interpolated period endpoint')
		plt.legend()
	else:
		_, _, _, current_xdot, current_zdot, \
			current_t = my_system.analytical_period_info(x, z, t)
		plt.subplot(211)
		plt.scatter(omega * np.array(current_t), np.array(current_xdot) / U,
					c='k', zorder=4, label='chosen period endpoint')
		plt.legend()

		plt.subplot(212)
		plt.scatter(omega * np.array(current_t), np.array(current_zdot) / U,
					c='k', zorder=4, label='chosen period endpoint')
		plt.legend()

def plot_particle_trajectory(my_system, x, z, xdot, zdot, t):
	k = my_system.get_wavenum()
	_, _, _, _, _, _, numerical_x, numerical_z, \
	   _, _, _ = my_system.numerical_period_info(x, z, xdot, zdot, t)
	_, analytical_x, analytical_z, \
	_, _, _ = my_system.analytical_period_info(x, z, t)

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

def plot_drift_velocity(my_system, x, z, xdot, zdot, t, delta_t,
						limit_axes):
	U = my_system.get_max_velocity()
	omega = my_system.get_angular_freq()
	analytical_ud, analytical_wd = my_system.analytical_drift_velocity(t=t)
	my_ud, my_wd = my_system.my_analytical_drift_velocity(t=t)
	u_d, w_d, numerical_t = my_system.numerical_drift_velocity(x, z, xdot,
																   zdot, t)
	xdot_avg, zdot_avg, \
		analytical_t = my_system.analytical_averages(x, z, t)

	plt.suptitle('Drift Velocity Comparison with '
				 + r'$\Delta t =$ {:.0e}'.format(delta_t), fontsize=16)
	plt.subplot(121)
	plt.xlabel(r'$ \omega t $', fontsize=12)
	plt.ylabel(r'$ u_d / U $', fontsize=12)
	plt.scatter(omega * analytical_t, xdot_avg / U, zorder=4, c='k', marker='x',
				label='average of eq (11) from Santamaria 2013')
	plt.plot(omega * t, analytical_ud / U, c='mediumpurple', zorder=1, 
			 label='eq (13) as listed in Santamaria 2013')
	plt.plot(omega * t, my_ud / U, c='k', zorder=2,
			 label='eq (13) from our calculations')
	plt.scatter(omega * numerical_t, u_d / U, zorder=3, 
				facecolors='none', edgecolors='k', label='numerical')
	plt.legend()

	plt.subplot(122)
	plt.xlabel(r'$ \omega t $', fontsize=12)
	plt.ylabel(r'$ w_d / U $', fontsize=12)
	plt.axhline(my_system.get_settling_velocity() / U, c='k', linestyle=':')
	plt.scatter(omega * analytical_t, zdot_avg / U, zorder=4, c='k', marker='x',
				label='average of eq (12) from Santamaria 2013')
	plt.plot(omega * t, analytical_wd / U, c='mediumpurple', zorder=1,
			 label='eq (14) as listed in Santamaria 2013')
	plt.plot(omega * t, my_wd / U, c='k', zorder=2,
			 label='eq (14) from our calculations')
	plt.scatter(omega * numerical_t, w_d / U, zorder=3, 
				facecolors='none', edgecolors='k', label='numerical')
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

def plot_drift_velocity_varying_delta_t(my_system, x, z, xdot, zdot, t,
										limit_axes):
	U = my_system.get_max_velocity()
	omega = my_system.get_angular_freq()

	# coarse delta t
	x_coarse, z_coarse, xdot_coarse, zdot_coarse, \
		t_coarse = my_system.particle_trajectory(santamaria, delta_t=5e-3)
	numerical_xdot_coarse, numerical_zdot_coarse, \
		numerical_t_coarse = my_system.numerical_drift_velocity(x_coarse, 
								 z_coarse, xdot_coarse, zdot_coarse, t_coarse)
	# medium delta t
	analytical_ud, analytical_wd = my_system.analytical_drift_velocity(t=t)
	numerical_xdot_med, numerical_zdot_med, \
		numerical_t_med = my_system.numerical_drift_velocity(x, z, xdot,
																 zdot, t)
	# fine delta t
	x_fine, z_fine, xdot_fine, zdot_fine, \
		t_fine = my_system.particle_trajectory(santamaria, delta_t=5e-7)
	numerical_xdot_fine, numerical_zdot_fine, \
		numerical_t_fine = my_system.numerical_drift_velocity(x_fine,
							   z_fine, xdot_fine, zdot_fine, t_fine)
	
	plt.suptitle(r'Drift Velocity Comparison with Varying $\Delta t$',
				 fontsize=16)
	plt.subplot(121)
	plt.xlabel(r'$ \omega t $', fontsize=12)
	plt.ylabel(r'$ u_d / U $', fontsize=12)
	plt.plot(omega * t, analytical_ud / U, c='k', zorder=1, label='analytical')
	plt.scatter(omega * numerical_t_coarse, numerical_xdot_coarse / U,
				zorder=2, marker='s', facecolors='none', edgecolors='k',
				label=r'coarse $\Delta t$')
	plt.scatter(omega * numerical_t_med, numerical_xdot_med / U, zorder=2, 
				facecolors='none', edgecolors='k', label=r'medium $\Delta t$')
	plt.scatter(omega * numerical_t_fine, numerical_xdot_fine / U, zorder=4,
				c='k', marker='x', label=r'fine $\Delta t$')
	plt.legend()

	plt.subplot(122)
	plt.xlabel(r'$ \omega t $', fontsize=12)
	plt.ylabel(r'$ w_d / U $', fontsize=12)
	plt.axhline(my_system.get_settling_velocity() / U, c='k', linestyle=':')
	plt.plot(omega * t, analytical_wd / U, c='k', zorder=1, label='analytical')
	plt.scatter(omega * numerical_t_coarse,
				numerical_zdot_coarse / U, zorder=2,
				marker='s', facecolors='none', edgecolors='k',
				label=r'coarse $\Delta t$')
	plt.scatter(omega * numerical_t_med, numerical_zdot_med / U, zorder=3,
				facecolors='none', edgecolors='k', label=r'medium $\Delta t$')
	plt.scatter(omega * numerical_t_fine, numerical_zdot_fine / U,
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

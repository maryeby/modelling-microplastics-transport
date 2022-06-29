import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import constants

def main():
	# define variables
	amplitude = 0.3						# A
	wavelength = 100 					# lambda
	wave_num = 2 * np.pi / wavelength 	# k
	depth = 500 						# h
	# calculate omega using dispersion relation
	angular_freq = np.sqrt(constants.g * wave_num * np.tanh(wave_num * depth))
	period = 2 * np.pi / angular_freq	# period of particle oscillation
	density = 2 / 3						# R
	stokes_num = 0.0656					# St

	# create figure and plots
	plt.figure()	
	particle_orbits(wave_num, depth, angular_freq, amplitude, period, density,
					stokes_num)	
	compare_drift_velocities(wave_num, depth, angular_freq, amplitude, period,
							 density, stokes_num)	
	plt.show()

def velocity_field(amplitude, angular_freq, wave_num, x, z, t):
	"""
	Returns the velocity field vector u for deep water.

	Parameters
	----------
	amplitude : float
		The amplitude of the wave, denoted A in the corresponding mathematics.
	angular_freq : float
		The angular frequency, denoted omega in the corresponding mathematics.
	wave_num : float
		The wave number, denoted k in the corresponding mathematics.
	x : float
		The x position at which to evaluate the velocity.
	z : float
		The z position at which to evaluate the velocity.
	t : float
		The time at which to evaluate the velocity.

	Returns
	-------
	NumPy array containing the velocity field vector components.
	"""
	return np.array([amplitude * angular_freq * np.exp(wave_num * z) 
							   * np.sin(angular_freq * t - wave_num * x),
					 amplitude * angular_freq * np.exp(wave_num * z)
							   * np.cos(angular_freq * t - wave_num * x)])

def particle_orbits(wave_num, depth, angular_freq, amplitude, period, density,
					stokes_num):
	"""
	Computes and plots the particle orbits for various initial depths.

	Parameters
	----------
	wave_num : float
		The wave number, denoted k in the corresponding mathematics.
	depth : float
		The depth of the water, denoted h in the corresponding mathematics.
	angular_freq : float
		The angular frequency, denoted omega in the corresponding mathematics.
	amplitude : float
		The amplitude of the wave, denoted A in the corresponding mathematics.
	period : float
		The period of the particle oscillations.
	density : float
		The density, denoted R in the corresponding mathematics.
	stokes_num : float
		The Stokes number, denoted St in the corresponding mathematics.
	"""
	# intialize local variables
	t_span = (0, 10 * period)			# time span
	initial_depths = range(-1, -5, -1)	# list of z_0 values
	x_0, u_0, w_0 = 0, 1, 0				# initial horizontal and velocity

	# compute solutions for each initial depth
	for z_0 in initial_depths:
		x, z, u, w = integrate.solve_ivp(maxey_riley, t_span,
										 [x_0, z_0, u_0, w_0], method='RK23',
								  		 args=(wave_num, depth, angular_freq,
											   amplitude, density,
											   stokes_num)).y
		# plot solution
		plt.subplot(121)
		plt.plot(x, z, 'k')

	# add attributes to plot
	plt.title('Particle Trajectory for Deep Water')
	plt.xlabel('Horizontal x')
	plt.ylabel('Depth z')

def compare_drift_velocities(wave_num, depth, angular_freq, amplitude, period,
							 density, stokes_num):
	"""
	Compares and plots the drift velocities for various initial depths.

	Parameters
	----------
	wave_num : float
		The wave number, denoted k in the corresponding mathematics.
	depth : float
		The depth of the water, denoted h in the corresponding mathematics.
	angular_freq : float
		The angular frequency, denoted omega in the corresponding mathematics.
	amplitude : float
		The amplitude of the wave, denoted A in the corresponding mathematics.
	period : float
		The period of the particle oscillations.
	density : float
		The density, denoted R in the corresponding mathematics.
	stokes_num : float
		The Stokes number, denoted St in the corresponding mathematics.
	"""
	# initialize local variables
	analytical_drift_vels = []
	numerical_drift_vels = []
	phase_speed = angular_freq / wave_num	# c
	t_span = (0, 10 * period)				# time span
	initial_depths = range(-1, -11, -1) 	# list of z_0 values
	x_0 = 0

	for z_0 in initial_depths:
		# make the initial particle velocity the same as the fluid velocity
		u_0, w_0 = velocity_field(amplitude, angular_freq, wave_num,
										   x_0, z_0, 0)
	
		# compute numerical solutions for each initial depth
		x, z, u, w = integrate.solve_ivp(maxey_riley, t_span,
											[x_0, z_0, u_0, w_0],
											method='RK23',
											args=(wave_num, depth,
												  angular_freq, amplitude,
												  density, stokes_num)).y
		
		# compute numerical and analytical drift velocities and add the results
		# to their corresponding list
		numerical_drift_vels.append((x[-1] - x[0]) / t_span[1])
		analytical_drift_vels.append(phase_speed * (amplitude * wave_num) ** 2 
												 * np.exp(2 * wave_num * z_0))
		
	# plot results
	plt.subplot(122)
	plt.plot(analytical_drift_vels, initial_depths, 'm--',
			 label='Analytical solution')
	plt.scatter(numerical_drift_vels, initial_depths, c='k', marker='^',
				label='Numerical solution')
	plt.title('Drift Velocity Comparison for Deep Water')
	plt.xlabel(r'Drift velocity $ u_d $')
	plt.legend()


def maxey_riley(t, y, wave_num, depth, angular_freq, amplitude, density,
				stokes_num):
	"""
	Returns the evaluation of the M-R equation at a specific time and position.

	Parameters
    ----------
	t : float
		The time to use in the computation.
	y : list (array-like)
		A list containing the x, z, u, and w values to use.
	wave_num : float
		The wave number, denoted k in the corresponding mathematics.
	depth : float
		The depth of the water, denoted h in the corresponding mathematics.
	angular_freq : float
		The angular frequency, denoted omega in the corresponding mathematics.
	amplitude : float
		The amplitude of the wave, denoted A in the corresponding mathematics.
	density : float
		The density, denoted R in the corresponding mathematics.
	stokes_num : float
		The Stokes number, denoted St in the corresponding mathematics.
	
	Returns
	-------
	NumPy array
		An array containing the x and z components of the particle's
		velocity and the x and z components of the particle's acceleration.
		These are denoted by  u, w, du/dt, and dw/dt in the corresponding
		mathematics.
	"""
	x, z = y[:2]				# current position
	particle_velocity = y[2:]	# u and w components of particle velocity
	fluid_velocity = velocity_field(amplitude, angular_freq, wave_num, x, z, t)
	
	# full derivative along the trajectory of the fluid element, Du / Dt
	fluid_derivative = np.array([amplitude * angular_freq * angular_freq
										   * np.exp(wave_num * z)
										   * np.cos(angular_freq * t
													- wave_num * x),
								 amplitude * angular_freq * angular_freq
										   * np.exp(wave_num * z)
										   * np.sin(angular_freq * t
													- wave_num * x)]) \
					   + np.dot(fluid_velocity, np.gradient(fluid_velocity))

	# compute the terms of the Maxey-Riley equation (neglecting history)
	fluid_pressure_gradient = 3 / 2 * density * fluid_derivative
	added_mass_force = (1 - 3 / 2 * density) * np.array([0, -constants.g])
	stokes_drag = density / stokes_num * (particle_velocity - fluid_velocity)

	# particle acceleration is the LHS of the M-R equation, denoted dv/dt
	particle_acceleration = fluid_pressure_gradient + added_mass_force \
													- stokes_drag

	return np.concatenate((particle_velocity, particle_acceleration))

main()

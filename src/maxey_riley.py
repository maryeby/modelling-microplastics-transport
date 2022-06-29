import numpy as np
import scipy.integrate as integrate
from scipy import constants

def velocity_field(amplitude, angular_freq, wave_num, depth, x, z, t):
	"""
	Returns the velocity field vector u for water of arbitrary depth.

	Parameters
	----------
	amplitude : float
		The amplitude of the wave, denoted A in the corresponding mathematics.
	angular_freq : float
		The angular frequency, denoted omega in the corresponding mathematics.
	wave_num : float
		The wave number, denoted k in the corresponding mathematics.
	depth : float
		The depth of the water, denoted h in the corresponding mathematics.
	x : float
		The x position at which to evaluate the velocity.
	z : float
		The z position at which to evaluate the velocity.
	t : float
		The time at which to evaluate the velocity.

	Returns
	-------
	Array containing the velocity field vector components.
	"""
	return np.array([amplitude * angular_freq * np.cosh(wave_num * (z + depth))
							   * np.sin (angular_freq * t - wave_num * x)
							   / np.sinh(wave_num * depth),
					 amplitude * angular_freq * np.sinh(wave_num * (z + depth))
							   * np.cos (angular_freq * t - wave_num * x)
							   / np.sinh(wave_num * depth)])

def particle_trajectory(x_0, z_0, u_0, w_0, period, wave_num, depth,
						angular_freq, amplitude, density, stokes_num):
	"""
	Computes the particle trajectory for specified initial conditions.

	Parameters
	----------
	x_0 : float
		The initial horizontal position of the particle.
	z_0 : float
		The initial vertical position of the particle.
	u_0 : float
		The initial horizontal velocity of the particle.
	w_0 : float
		The initial vertical velocity of the particle.
	period : float
		The period of the particle oscillations.
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
	x : array
		An array of x values representing the horizontal particle position.
	z : array
		An array of z values representing the vertiacal particle position.
	"""
	t_span = (0, 20 * period)	# time span

	# compute numerical solutions
	x, z, u, w = integrate.solve_ivp(mr_no_history, t_span,
									 [x_0, z_0, u_0, w_0], method='BDF',
							  		 args=(wave_num, depth, angular_freq,
										   amplitude, density,
										   stokes_num)).y
	return x, z	

def compare_drift_velocities(initial_depths, x_0, period, wave_num, depth,
							 angular_freq, amplitude, density, stokes_num):
	"""
	Compares and plots the drift velocities for various initial depths.

	Parameters
	----------
	initial_depths : array
		An array of z_0 values representing the initial depths.
	x_0 : float
		The particle's initial horizontal position.
	period : float
		The period of the particle oscillations.
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
	"""
	# initialize local variables
	analytical_drift_vels = []
	numerical_drift_vels = []
	phase_speed = angular_freq / wave_num	# c
	t_span = (0, 20 * period)				# time span

	for z_0 in initial_depths:
		# make the initial particle velocity the same as the fluid velocity
		u_0, w_0 = velocity_field(amplitude, angular_freq, wave_num, depth,
										   x_0, z_0, 0)
	
		# run numerics for each initial depth
		x, z, u, w = integrate.solve_ivp(mr_no_history, t_span,
											[x_0, z_0, u_0, w_0],
											method='BDF',
											args=(wave_num, depth,
												  angular_freq, amplitude,
												  density, stokes_num)).y
		
		# compute numerical and analytical drift velocities
		numerical_sol = (x[-1] - x[0]) / t_span[1]
		analytical_sol = phase_speed * (amplitude * wave_num) ** 2 \
									 * np.cosh(2 * wave_num * (z_0 + depth)) \
									 / (2 * np.sinh(wave_num * depth) ** 2)

		# add the solutions to their corresponding list
		numerical_drift_vels.append(numerical_sol)
		analytical_drift_vels.append(analytical_sol)
		
	return np.array(numerical_drift_vels), np.array(analytical_drift_vels)

def mr_no_history(t, y, wave_num, depth, angular_freq, amplitude, density,
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
	fluid_velocity = velocity_field(amplitude, angular_freq, wave_num, depth,
									x, z, t)
	
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
	buoyancy_force = (1 - 3 / 2 * density) * np.array([0, -constants.g])
	stokes_drag = density / stokes_num * (particle_velocity - fluid_velocity)

	# particle acceleration is the LHS of the M-R equation, denoted dv/dt
#	particle_acceleration = fluid_pressure_gradient + buoyancy_force \
#													- stokes_drag
	particle_acceleration = -stokes_drag

	return np.concatenate((particle_velocity, particle_acceleration))

def mr_with_history(t, y, wave_num, depth, angular_freq, amplitude, density,
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
	fluid_velocity = velocity_field(amplitude, angular_freq, wave_num, depth,
									x, z, t)
	store_history(particle_velocity, fluid_velocity)	

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
	buoyancy_force = (1 - 3 / 2 * density) * np.array([0, -constants.g])
	stokes_drag = density / stokes_num * (particle_velocity - fluid_velocity)
	history = history()

	# particle acceleration is the LHS of the M-R equation, denoted dv/dt
	particle_acceleration = fluid_pressure_gradient + buoyancy_force \
													- stokes_drag - history

	return np.concatenate((particle_velocity, particle_acceleration))

def history(particle_vel_history, fluid_vel_history, time, density, stokes_num):
	"""
	Computes the value of the history term from the Maxey-Riley equation.
	
	Parameters
	----------
	particle_vel_history : array
		The history of the particle velocity for both the x and z components.
	fluid_vel_history : array
		The history of the fluid velocity for both the x and z components.
	time : array
		The time steps over which the ODE solver will be iterating.
	density : float
		The density, denoted R in the corresponding mathematics.
	stokes_num : float
		The Stokes number, denoted St in the corresponding mathematics.
	"""
	coefficient = np.sqrt(9 / (2 * np.pi)) * (density / np.sqrt(stokes_num))

	for i in range(len(timesteps) - 1):
		delta_t = time[i + 1] - time[i]
		result += delta_t / (time[-1] - time[i]) \
						  * (particle_vel_history - fluid_vel_history)

	return coefficient * result

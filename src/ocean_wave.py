import numpy as np
import scipy.integrate as integrate
from scipy import constants

class OceanWave:
	"""
	Represent a wave in the ocean.

	Attributes
	----------
	amplitude : float
		The amplitude of the wave, denoted A in the corresponding mathematics.
	wavelength : float
		The wavelength of the wave, denoted lambda in corresponding mathematics.
	depth : float
		The depth of the water, denoted h in the corresponding mathematics.
	density : float
		The density, denoted R in the corresponding mathematics.
	stokes_num : float
		The Stokes number, denoted St in the corresponding mathematics.
	wave_num : float
		The wave number, denoted k in the corresponding mathematics.
	angular_freq : float
		The angular frequency, denoted omega in the corresponding mathematics.
	period : float
		The period of the particle oscillations.
	particle_vel_history : array
		An array containing the previous particle velocity values.
	fluid_vel_history : array
		An array containing the previous fluid velocity values.
	timesteps : array
		An array containing the time steps where the M-R equation is evaluated.
	"""
	
	def __init__(self, amplitude=0.1, wavelength=10, depth=8, density=2/3,
				 stokes_num=0.1, beta=1):
		"""Create an OceanWave object."""
		self.__amplitude = amplitude	# A
		self.__wavelength = wavelength	# lambda
		self.__depth = depth			# h
		self.__density = density		# R
		self.__stokes_num = stokes_num	# St
		self.__beta = beta

		self.__wave_num = 2 * np.pi / self.__wavelength	# k
		self.__angular_freq = np.sqrt(constants.g * self.__wave_num
												  * np.tanh(self.__wave_num
												  * self.__depth))		# omega
		self.__max_velocity = self.__angular_freq * self.__amplitude	# U
		self.__response_time = self.__stokes_num / self.__angular_freq	# tau

		self.__period = 2 * np.pi / self.__angular_freq	# period of oscillation
		self.__particle_history = []	# history of v dot
		self.__fluid_history = []		# history of u dot
		self.__timesteps = []			# time steps where M-R is evaluated

	def get_amplitude(self):
		"""Return the amplitude A."""
		return self.__amplitude

	def get_wavelength(self):
		"""Return the wavelength lamda."""
		return self.__wavelength

	def get_depth(self):
		"""Return the depth h."""
		return self.__depth

	def get_density(self):
		"""Return the density R."""
		return self.__density

	def get_stokes_num(self):
		"""Return the Stokes number St."""
		return self.__stokes_num

	def get_beta(self):
		"""Return the value of beta."""
		return self.__beta

	def get_wave_num(self):
		"""Return the wave number k."""
		return self.__wave_num

	def get_angular_freq(self):
		"""Return the angular frequency omega."""
		return self.__angular_freq

	def get_max_velocity(self):
		"""Return the maximum velocity U."""
		return self.__angular_freq

	def get_respnse_time(self):
		"""Return the response time tau."""
		return self.__response_time

	def get_period(self):
		"""Return the period."""
		return self.__period

	def get_particle_vel_history(self):
		"""Return the particle velocity history."""
		return self.__particle_vel_history

	def get_fluid_vel_history(self):
		"""Return the fluid velocity history."""
		return self.__fluid_vel_history

	def get_timesteps(self):
		"""Return the time steps."""
		return self.__timesteps

	def fluid_velocity(self, x, z, t):
		"""
		Returns the fluid velocity vector u for water of arbitrary depth.

		Parameters
		----------
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
		return np.array([0,0])
#		return np.array([self.__amplitude * self.__angular_freq
#										  * np.cosh(self.__wave_num
#													* (z + self.__depth))
#										  * np.sin(self.__angular_freq * t 
#												   - self.__wave_num * x)
#										  / np.sinh(self.__wave_num
#													* self.__depth),
#						 self.__amplitude * self.__angular_freq
#										  * np.sinh(self.__wave_num
#													* (z + self.__depth))
#										  * np.cos(self.__angular_freq * t
#												   - self.__wave_num * x)
#										  / np.sinh(self.__wave_num
#													* self.__depth)])

	def fluid_accel(self, x, z, t):
		"""
		Returns the fluid acceleration u dot for water of arbitrary depth.

		Parameters
		----------
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
		return np.array([0,0])
#		return np.array([self.__amplitude * self.__angular_freq ** 2
#										  * np.cosh(self.__wave_num
#													* (z + self.__depth))
#										  * np.cos(self.__angular_freq * t 
#												   - self.__wave_num * x)
#										  / np.sinh(self.__wave_num
#													* self.__depth),
#						 -self.__amplitude * self.__angular_freq ** 2
#										   * np.sinh(self.__wave_num
#													 * (z + self.__depth))
#										   * np.cos(self.__angular_freq * t
#												    - self.__wave_num * x)
#										   / np.sinh(self.__wave_num
#													 * self.__depth)])

	def fluid_derivative(self, x, z, t):
		"""
		Returns the derivative along the trajectory of the fluid element, Du/Dt.

		Parameters
		----------
		x : float
			The x position at which to evaluate the velocity.
		z : float
			The z position at which to evaluate the velocity.
		t : float
			The time at which to evaluate the velocity.

		Returns
		-------
		Array containing the fluid derivative vector components.
		"""
		fluid_velocity = self.fluid_velocity(x, z, t)
		return np.array([self.__amplitude * self.__angular_freq ** 2
										  * np.cosh(self.__wave_num
													* (z + self.__depth))
										  * np.cos(self.__angular_freq * t
												   - self.__wave_num * x)
										  / np.sinh(self.__wave_num
													* self.__depth),
						 self.__amplitude * self.__angular_freq ** 2
										  * np.sinh(self.__wave_num
													* (z + self.__depth))
										  * np.sin(self.__angular_freq * t
												   - self.__wave_num * x)
										  / np.sinh(self.__wave_num
													* self.__depth)]) \
						 + np.dot(fluid_velocity, np.gradient(fluid_velocity))

	def particle_trajectory(self, fun, x_0, z_0, u_0, w_0):
		"""
		Computes the particle trajectory for specified initial conditions.

		Parameters
		----------
		fun : function
			The name of the function to use to evaluate the M-R equation.
		x_0 : float
			The initial horizontal position of the particle.
		z_0 : float
			The initial vertical position of the particle.
		u_0 : float
			The initial horizontal velocity of the particle.
		w_0 : float
			The initial vertical velocity of the particle.

		Returns
		-------
		x : array
			The horizontal particle positions.
		z : array
			The vertical particle positions.
		"""
		t_span = (0, 20 * self.__period + 0.1 * self.__period)	# time span
		x, z, _, _ = integrate.solve_ivp(fun, t_span, [x_0, z_0, u_0, w_0],
								   method='BDF', rtol=1e-8, atol=1e-10).y
		return x, z

	def particle_velocity(self, fun, x_0, z_0, u_0, w_0):
		"""
		Computes the particle velocities for specified initial conditions.

		Parameters
		----------
		fun : function
			The name of the function to use to evaluate the M-R equation.
		x_0 : float
			The initial horizontal position of the particle.
		z_0 : float
			The initial vertical position of the particle.
		u_0 : float
			The initial horizontal velocity of the particle.
		w_0 : float
			The initial vertical velocity of the particle.

		Returns
		-------
		u : array
			The horizontal particle velocities.
		w : array
			The vertical particle velocities.
		"""
		t_span = (0, 20 * self.__period + 0.1 * self.__period)	# time span
		sols = integrate.solve_ivp(fun, t_span, [x_0, z_0, u_0, w_0],
								   method='BDF', rtol=1e-8, atol=1e-10)
		_, _, u, w = sols.y
		time = sols.t
		return u, w, time

	def compare_drift_velocities(self, fun, initial_depths, x_0):
		"""
		Compares and plots the drift velocities for various initial depths.

		Parameters
		----------
		fun : function
			The name of the function to use to evaluate the M-R equation.
		initial_depths : array
			An array of z_0 values representing the initial depths.
		x_0 : float
			The particle's initial horizontal position.
		"""
		# initialize local variables
		analytical_drift_vels = []
		numerical_drift_vels = []
		phase_speed = self.__angular_freq / self.__wave_num		# c
		t_span = (0, 20 * self.__period + self.__period * 0.1)	# time span

		for z_0 in initial_depths:
			# make the initial particle velocity the same as the fluid velocity
			u_0, w_0 = self.fluid_velocity(x_0, z_0, 0)
		
			# run numerics for each initial depth
			x, z, u, w = integrate.solve_ivp(fun, t_span, [x_0, z_0, u_0, w_0],
											 method='BDF', rtol=1e-8,
											 atol=1e-10).y
			
			# find where the trajectory completes the last orbit
			if z_0 < z[-1]:
				index = np.where(z <= z_0)[-1][-1]
			elif z[-1] < z_0:
				index = np.where(z >= z_0)[-1][-1]
			else:
				index = len(z) - 1

			if z[index] == z_0:
				x_final = x[index]
			else:
				x_final = (x[index] + x[index + 1]) / 2
	
			# compute numerical and analytical drift velocities
			numerical_sol = (x_final - x_0) / t_span[1]
			analytical_sol = phase_speed * (self.__amplitude * self.__wave_num)\
										 ** 2 * np.cosh(2 * self.__wave_num
														* (z_0 + self.__depth))\
										 / (2 * np.sinh(self.__wave_num
														* self.__depth) ** 2)

			# add the solutions to their corresponding list
			numerical_drift_vels.append(numerical_sol)
			analytical_drift_vels.append(analytical_sol)
			
		return np.array(numerical_drift_vels), np.array(analytical_drift_vels)

	def mr_no_history(self, t, y):
		"""
		Returns the evaluation of the M-R equation without the history term.

		Parameters
		----------
		t : float
			The time to use in the computation.
		y : list (array-like)
			A list containing the x, z, u, and w values to use.
		
		Returns
		-------
		array
			An array containing the x and z components of the particle's
			velocity and the x and z components of the particle's acceleration.
			These are denoted by  u, w, du/dt, and dw/dt in the corresponding
			mathematics.
		"""
		x, z = y[:2]				# current position
		particle_velocity = y[2:]	# u and w components of particle velocity

		# compute the terms of the Maxey-Riley equation (neglecting history)
		fluid_pressure_gradient = 3 / 2 * self.__density \
										* self.fluid_derivative(x, z, t)
		buoyancy_force = (1 - 3 / 2 * self.__density) \
						 	* np.array([0, -constants.g])
		stokes_drag = self.__density / self.__stokes_num * (particle_velocity 
									 - self.fluid_velocity(x, z, t))

		# particle acceleration is the LHS of the M-R equation, denoted dv/dt
		particle_acceleration = fluid_pressure_gradient + buoyancy_force \
														- stokes_drag

		return np.concatenate((particle_velocity, particle_acceleration))

	def santamaria(self, t, y):
		"""
		Returns the evaluation of the M-R equation using Santamaria's model.

		Parameters
		----------
		t : float
			The time to use in the computation.
		y : list (array-like)
			A list containing the x, z, u, and w values to use.
		
		Returns
		-------
		NumPy array
			An array containing the x and z components of the particle's
			velocity and the x and z components of the particle's acceleration.
			These are denoted by  u, w, du/dt, and dw/dt in the corresponding
			mathematics.
		"""	
		x, z = y[:2]	# current position
		particle_velocity = y[2:]	# u and w components of particle velocity
		fluid_velocity = np.array([self.__max_velocity
								   * np.exp(self.__wave_num * z)
								   * np.cos(self.__wave_num * x
											- self.__angular_freq * t),
						  		   self.__max_velocity
								   * np.exp(self.__wave_num * z)
								   * np.sin(self.__wave_num * x
											- self.__angular_freq * t)])
		fluid_acceleration = np.array([self.__angular_freq * self.__max_velocity
									   * np.exp(self.__wave_num * z)
									   * np.sin(self.__wave_num * x
												- self.__angular_freq * t),
						  	  		   -self.__angular_freq
									   * self.__max_velocity
									   * np.exp(self.__wave_num * z)
									   * np.cos(self.__wave_num * x
												- self.__angular_freq * t)])
		if t == 0:
			particle_velocity = fluid_velocity

		fluid_pressure_gradient = self.__beta * fluid_acceleration 
		buoyancy_force = (1 - self.__beta) * np.array([0, -constants.g])
		stokes_drag = (fluid_velocity - particle_velocity) \
									  / self.__response_time

		# particle acceleration is the LHS of the M-R equation, denoted dv/dt
		particle_acceleration = fluid_pressure_gradient + buoyancy_force \
														+ stokes_drag

		return np.concatenate((particle_velocity, particle_acceleration))

	def mr_with_history(self, t, y):
		"""
		Returns the evaluation of the M-R equation with the history term.

		Parameters
		----------
		t : float
			The time to use in the computation.
		y : list (array-like)
			A list containing the x, z, u, and w values to use.
		
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

		# compute the terms of the Maxey-Riley equation except history
		fluid_pressure_gradient = 3 / 2 * self.__density \
										* self.fluid_derivative(x, z, t)
		buoyancy_force = (1 - 3 / 2 * self.__density) \
						 * np.array([0, -constants.g])
		stokes_drag = self.__density / self.__stokes_num * (particle_velocity
									 - self.fluid_velocity(x, z, t))

		# compute the history term
		integrand = np.empty(2,)
		coefficient = np.sqrt(9 / (2 * np.pi)) \
					  * (self.__density / np.sqrt(self.__stokes_num))
		kernel = np.array(self.__particle_history) \
				 - np.array(self.__fluid_history)
		num_steps = len(self.__timesteps) - 1
		alpha = 0
		delta_t = 0

		for j in range(num_steps):
			if j == 0:
				alpha = 4 / 3
			elif j == num_steps:
				alpha = (4 / 3) * (num_steps - 1) ** (3 / 2) \
								- num_steps ** (3 / 2) + (3 / 2) \
								* np.sqrt(num_steps)
				delta_t = np.abs(self.__timesteps[j] - self.__timesteps[j - 1])
			else:
				alpha = (4 / 3) * (j - 1) ** (3 / 2) + (j + 1) ** (3 / 2) - 2 \
								* j ** (3 / 2) 
				delta_t = np.abs(self.__timesteps[j] - self.__timesteps[j - 1])
			integrand += np.sqrt(delta_t) * alpha * kernel[num_steps - j]
		history = coefficient * integrand

		# particle acceleration is the LHS of the M-R equation, denoted dv/dt
		particle_acceleration = fluid_pressure_gradient + buoyancy_force \
														- stokes_drag - history
		# update relevant OceanWave attributes
		self.__particle_history.append(particle_acceleration)
		self.__fluid_history.append(self.fluid_accel(x, z, t))
		self.__timesteps.append(t)

		return np.concatenate((particle_velocity, particle_acceleration))

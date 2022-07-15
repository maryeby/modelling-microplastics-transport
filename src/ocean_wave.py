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
				 stokes_num=0.1):
		"""Create an OceanWave object."""
		self.__amplitude = amplitude	# A
		self.__wavelength = wavelength	# lambda
		self.__depth = depth			# h
		self.__density = density		# R
		self.__stokes_num = stokes_num	# St
		self.__wave_num = 2 * np.pi / self.__wavelength	# k
		# omega calculated using the dispersion relation
		self.__angular_freq = np.sqrt(constants.g * self.__wave_num
												  * np.tanh(self.__wave_num
															* self.__depth))
		self.__period = 2 * np.pi / self.__angular_freq	# period of oscillation
		self.__particle_vel_history = []	# history of vector v
		self.__fluid_vel_history = []		# history of vector u
		self.__timesteps = []				# time steps where M-R is evaluated

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

	def get_wave_num(self):
		"""Return the wave number k."""
		return self.__wave_num

	def get_angular_freq(self):
		"""Return the angular frequency omega."""
		return self.__angular_freq

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

	def velocity_field(self, x, z, t):
		"""
		Returns the velocity field vector u for water of arbitrary depth.

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
		return np.array([self.__amplitude * self.__angular_freq
										  * np.cosh(self.__wave_num
													* (z + self.__depth))
										  * np.sin(self.__angular_freq * t 
												   - self.__wave_num * x)
										  / np.sinh(self.__wave_num
													* self.__depth),
						 self.__amplitude * self.__angular_freq
										  * np.sinh(self.__wave_num
													* (z + self.__depth))
										  * np.cos(self.__angular_freq * t
												   - self.__wave_num * x)
										  / np.sinh(self.__wave_num
													* self.__depth)])

	def particle_trajectory(self, x_0, z_0, u_0, w_0):
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

		Returns
		-------
		x : array
			The horizontal particle positions computed without history.
		z : array
			The vertical particle positions computed without history.
		x_hist : array
			The horizontal particle positions computed with history.
		z_hist : array
			The vertical particle positions computed with history.
		"""
		t_span = (0, 100 * self.__period + 0.1 * self.__period)	# time span

		x, z, _, _ = integrate.solve_ivp(self.mr_no_history, t_span,
										 [x_0, z_0, u_0, w_0], method='BDF',
										 rtol=1e-8, atol=1e-10).y
		return x, z

	def compare_drift_velocities(self, initial_depths, x_0):
		"""
		Compares and plots the drift velocities for various initial depths.

		Parameters
		----------
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
			u_0, w_0 = self.velocity_field(x_0, z_0, 0)
		
			# run numerics for each initial depth
			x, z, u, w = integrate.solve_ivp(self.mr_no_history, t_span,
											 [x_0, z_0, u_0, w_0],
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
		fluid_velocity = self.velocity_field(x, z, t)

		# full derivative along the trajectory of the fluid element, Du / Dt
		fluid_derivative = np.array([
						   self.__amplitude * self.__angular_freq ** 2
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

		# compute the terms of the Maxey-Riley equation (neglecting history)
		fluid_pressure_gradient = 3 / 2 * self.__density * fluid_derivative
		buoyancy_force = (1 - 3 / 2 * self.__density) \
						 * np.array([0, -constants.g])
		stokes_drag = self.__density / self.__stokes_num \
									 * (particle_velocity - fluid_velocity)

		# particle acceleration is the LHS of the M-R equation, denoted dv/dt
		particle_acceleration = fluid_pressure_gradient + buoyancy_force \
														- stokes_drag

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
		timesteps : array
			The time steps over which the ODE solver will be iterating.
		
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
		fluid_velocity = self.velocity_field(x, z, t)

		# update relevant OceanWave attributes
		self.__particle_vel_history.append(particle_velocity)
		self.__fluid_vel_history.append(fluid_velocity)
		self.__timesteps.append(t)

		# full derivative along the trajectory of the fluid element, Du / Dt
		fluid_derivative = np.array([
						   self.__amplitude * self.__angular_freq ** 2
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

		# compute the terms of the Maxey-Riley equation except history
		fluid_pressure_gradient = 3 / 2 * self.__density * fluid_derivative
		buoyancy_force = (1 - 3 / 2 * self.__density) \
						 * np.array([0, -constants.g])
		stokes_drag = self.__density / self.__stokes_num \
									 * (particle_velocity - fluid_velocity)
#		history = self.history()

		# compute the history term
		integrand = np.empty(2,)
		coefficient = np.sqrt(9 / (2 * np.pi)) \
					  * (self.__density / np.sqrt(self.__stokes_num))

		if len(self.__timesteps) > 1: # ensures the array has 2+ elements
			for i in range(len(self.__timesteps) - 1):
				# avoid dividing by zero
				if self.__timesteps[-1] - self.__timesteps[i] != 0:
						delta_t = self.__timesteps[i + 1] - self.__timesteps[i]
						integrand += delta_t / (np.sqrt(
												self.__timesteps[-1]
											 	- self.__timesteps[i])) \
										  	 * (np.array(
											 	self.__particle_vel_history[i]) 
										  	 	- np.array(
											   	  self.__fluid_vel_history[i]))
		history = coefficient * integrand

		# particle acceleration is the LHS of the M-R equation, denoted dv/dt
		particle_acceleration = fluid_pressure_gradient + buoyancy_force \
														- stokes_drag - history

		return np.concatenate((particle_velocity, particle_acceleration))

	def history(self):
		"""Computes the value of the history term from the M-R equation."""
		result = np.empty(2,)
		# the coefficient of the history term
		coefficient = np.sqrt(9 / (2 * np.pi)) \
					  * (self.__density / np.sqrt(self.__stokes_num))

		if len(self.__timesteps) > 1: # ensures the array has 2+ elements
			for i in range(len(self.__timesteps) - 1):
				# avoid dividing by zero
				if self.__timesteps[-1] - self.__timesteps[i] != 0:
						delta_t = self.__timesteps[i + 1] - self.__timesteps[i]
				
						# compute the sum from 0 to t (corresponds to integrand)
						result += delta_t / (np.sqrt(np.abs(self.__timesteps[-1]
											 - self.__timesteps[i]))) \
										  * (np.array(
											 self.__particle_vel_history[i]) 
										  	 - np.array(
											   self.__fluid_vel_history[i]))

		return coefficient * result

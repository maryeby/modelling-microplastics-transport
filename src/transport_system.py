import numpy as np
import scipy.integrate as integrate
from scipy import constants

class TransportSystem:
	"""
	Represent the transport of an inertial particle in a linear ocean wave.

	Attributes
	----------
	amplitude : float, default=0.1
		The amplitude of the wave, denoted A in the corresponding mathematics.
	wavelength : float, default=10
		The wavelength of the wave, denoted lambda in corresponding mathematics.
	depth : float, default=8
		The depth of the water, denoted h in the corresponding mathematics.
	density : float, default=2/3
		The density, denoted R in the corresponding mathematics.
	stokes_num : float, default=1e-5
		The Stokes number, denoted St in the corresponding mathematics.
	beta : float, default=1
		The variable beta, related to the heaviness of the particle (SM 2013).
	wavenum : float
		The wave number, denoted k in the corresponding mathematics.
	angular_freq : float
		The angular frequency, denoted omega in the corresponding mathematics.
	max_velocity : float
		The maximum velocity U at the surface z=0 (Santamaria 2013).
	response_time : float
		The Stokes response time tau from Santamaria et al. 2013.
	period : float
		The period of the particle oscillations.
	particle_history : array
		An array containing the previous particle acceleration values (vdot).
	fluid_history : array
		An array containing the previous fluid acceleration values (udot).
	timesteps : array
		An array containing the time steps where the M-R equation is evaluated.

	Notes
	-----
	The parameters beta, max_velocity (U), and response_time (tau) are from
	Santamaria et al., 2013.
	"""

	def __init__(self, amplitude=0.1, wavelength=10, depth=8, density=2/3,
				 stokes_num=1e-5, beta=1):
		self.__amplitude = amplitude	# A
		self.__wavelength = wavelength	# lambda
		self.__depth = depth			# h
		self.__density = density		# R
		self.__stokes_num = stokes_num	# St
		self.__beta = beta

		self.__wavenum = 2 * np.pi / self.__wavelength	# k
		self.__angular_freq = np.sqrt(constants.g * self.__wavenum
												  * np.tanh(self.__wavenum
												  * self.__depth))	  # omega
		self.__max_velocity = self.__angular_freq * self.__amplitude	# U 
		self.__response_time = self.__stokes_num / self.__angular_freq  # tau
		self.__phase_velocity = self.__angular_freq / self.__wavenum	# c
		self.__settling_velocity = -(1 - self.__beta) * constants.g \
								   * self.__response_time
		self.__period = 2 * np.pi / self.__angular_freq # period of oscillation

		self.__particle_history = []	# history of v dot
		self.__fluid_history = []	   # history of u dot
		self.__timesteps = []		   # time steps where M-R is evaluated

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

	def get_wavenum(self):
		"""Return the wave number k."""
		return self.__wavenum

	def get_angular_freq(self):
		"""Return the angular frequency omega."""
		return self.__angular_freq

	def get_max_velocity(self):
		"""Return the maximum velocity U."""
		return self.__max_velocity

	def get_response_time(self):
		"""Return the response time tau."""
		return self.__response_time

	def get_phase_velocity(self):
		""" Return the phase velocity c."""
		return self.__phase_velocity
	
	def get_settling_velocity(self):
		"""Return the settling velocity."""
		return self.__settling_velocity

	def get_period(self):
		"""Return the period."""
		return self.__period

	def get_particle_history(self):
		"""Return the history of v dot."""
		return self.__particle_history

	def get_fluid_history(self):
		"""Return the history of u dot."""
		return self.__fluid_history

	def get_timesteps(self):
		"""Return the time steps."""
		return self.__timesteps

	def update_particle_history(self, new):
		"""Append a new value to the particle_history array."""
		self.__particle_history.append(new)

	def update_fluid_history(self, new):
		"""Append a new value to the fluid_history array."""
		self.__fluid_history.append(new)

	def update_timesteps(self, new):
		"""Append a new value to the timesteps array."""
		self.__timesteps.append(new)

	def set_stokes_num(self, stokes_num):
		"""Set the Stokes number to the specified value."""
		self.__stokes_num = stokes_num

	def fluid_velocity(self, x, z, t, deep=False, dimensional=False):
		"""
		Computes the fluid velocity vector u.

		Parameters
		----------
		x : float or array
			The x position(s) at which to evaluate the velocity.
		z : float or array
			The z position(s) at which to evaluate the velocity.
		t : float or array
			The time(s) at which to evaluate the velocity.
		deep : boolean, default=False
			Whether the water is assumed to be infinitely deep.
		dimensional : boolean, default=False
			Whether the expression should be dimensional.

		Returns
		-------
		Array containing the velocity field vector components.
		"""
		U = self.__max_velocity
		A = self.__amplitude
		k = self.__wavenum
		epsilon = A * k
		omega = self.__angular_freq
		h = self.__depth

		if deep and dimensional:
			return np.array([U * np.exp(k * z) * np.cos(k * x - omega * t),
							 U * np.exp(k * z) * np.sin(k * x - omega * t)])
		elif deep and not dimensional:
			return np.array([np.exp(z) * np.cos(x - t / epsilon),
							 np.exp(z) * np.sin(x - t / epsilon)])
		else:
			return np.array([A * omega * np.cosh(k * (z + h)) 
							   * np.sin(omega * t - k * x) / np.sinh(k * h),
							 A * omega * np.sinh(k * (z + h))
							   * np.cos(omega * t - k * x) / np.sinh(k * h)])

	def material_derivative(self, x, z, t, deep=False, dimensional=False):
		"""
		Computes the Lagrangian derivative, Du/Dt.

		Parameters
		----------
		x : float or array
			The x position(s) at which to evaluate the fluid velocity.
		z : float or array
			The z position(s) at which to evaluate the velocity and derivative.
		t : float or array
			The time(s) at which to evaluate the velocity.
		deep : boolean, default=False
			Whether the water is assumed to be infinitely deep.
		dimensional : boolean, default=False
			Whether the expression should be dimensional.

		Returns
		-------
		Array containing the material derivative vector components.
		"""
		U = self.__max_velocity
		k = self.__wavenum
		omega = self.__angular_freq
		epsilon = k * self.__amplitude # from Santamaria
		u, w = self.fluid_velocity(x, z, t, deep, dimensional)

		if dimensional:
			return np.array([omega * w,
							 k * U ** 2 * np.exp(2 * k * z) - omega * u])
		else:
			return np.array([w / epsilon, np.exp(2 * z) - u / epsilon])

	def material_derivative2(self, x, z, t, deep=False, dimensional=False):
		"""
		Computes the second order Lagrangian derivative.

		Parameters
		----------
		x : float or array
			The x position(s) at which to evaluate the fluid velocity.
		z : float or array
			The z position(s) at which to evaluate the velocity and derivative.
		t : float or array
			The time(s) at which to evaluate the velocity.
		deep : boolean, default=False
			Whether the water is assumed to be infinitely deep.
		dimensional : boolean, default=False
			Whether the expression should be dimensional.

		Returns
		-------
		Array containing the second order material derivative vector components.
		"""
		U = self.__max_velocity
		k = self.__wavenum
		omega = self.__angular_freq
		epsilon = k * self.__amplitude # from Santamaria
		u, w = self.fluid_velocity(x, z, t, deep, dimensional)

		if dimensional:
			return np.array([k * w * U ** 2 * np.exp(2 * k * z)
							   - omega ** 2 * u,
						 	 w * (2 * np.exp(2 * k * z) * U ** 2 * k ** 2 
							   - omega ** 2)])
		else:
			return np.array([np.exp(2 * z) - u / epsilon ** 2,
							 w * (2 * np.exp(2 * z) - 1 / epsilon ** 2)])

	def analytical_particle_velocity(self, x_0=0, z_0=0, t=0,
									 dimensional=False):
		"""
		Computes the analytical solutions for the particle velocity.

		Parameters
		----------
		x_0 : float, default=0
			The initial horizontal position of the particle.
		z_0 : float, default=0
			The initial vertical position of the particle.
		t : float or array, default=0
			The time(s) at which to evaluate the particle velocity.
		dimensional : boolean, default=False
			Whether the expression should be dimensional.

		Returns
		-------
		xdot : float or array
			The horizontal particle velocity.
		zdot : float or array
			The vertical particle velocity.

		Notes
		-----
		The formulas used to compute the dimensional analytical solutions are
		equations (11) and (12) from Santamaria et al., 2013.
		"""
		U = self.__max_velocity
		k = self.__wavenum
		St = self.__stokes_num
		c = self.__phase_velocity
		bprime = 1 - self.__beta
		epsilon = k * self.__amplitude
		z_0t = z_0 - St * bprime * t 

		if dimensional:
			phi = k * x_0 - self.__angular_freq * t
			xdot = U * np.exp(k * z_0t) * ((1 - St ** 2 * self.__beta * bprime)
					 * np.cos(phi) - St * bprime * np.sin(phi)) \
					 + U ** 2 / c * np.exp(2 * k * z_0t) \
					 * (1 - St ** 2 * self.__beta * bprime)
			zdot = U * np.exp(k * z_0t) * ((1 - St ** 2 * self.__beta * bprime)
					 * np.sin(phi) + St * bprime * np.cos(phi)) \
					 - c * St * bprime \
					 * (1 + 2 * U ** 2 / c ** 2 * np.exp(2 * k * z_0t))
		else:
			phi = x_0 - t
			xdot = np.exp(z_0t) * epsilon * (np.cos(phi) * (1 - St ** 2
								* bprime) - St * bprime * np.sin(phi)) \
								+ np.exp(2 * z_0t) * epsilon ** 2 * (1 - St ** 2
								* bprime)
			zdot = np.exp(z_0t) * epsilon * (np.sin(phi) * (1 - St ** 2
								* bprime) + St * bprime * np.cos(phi)) \
								- St * bprime * (1 + 2 * epsilon ** 2
													   * np.exp(2 * z_0t))
		return xdot, zdot

	def analytical_drift_velocity(self, x_0=0, z_0=0, t=0, shifted=False):
		"""
		Computes the analytical solutions for the drift velocity.

		Parameters
		----------
		x_0 : float, default=0
			The initial horizontal position of the particle.
		z_0 : float, default=0
			The initial vertical position of the particle.
		t : float or array, default=0
			The time(s) at which to evaluate the particle velocity.
		shifted : boolean, default=False
			Whether the analytical curve for w_d is shifted horizontally.

		Returns
		-------
		u_d : float or array
			The horizontal drift velocities.
		w_d : float or array
			The vertical drift velocities.

		Notes
		-----
		The formulas used to compute these analytical solutions are from
		Santamaria et al., 2013.
		"""
		U = self.__max_velocity
		k = self.__wavenum
		St = self.__stokes_num
		c = self.__phase_velocity
		bprime = 1 - self.__beta
		tau = self.__response_time

		# equation (13) from Santamaria 2013
		u_d = U ** 2 / c * (1 - self.__beta * bprime * St ** 2) \
				  * np.exp(2 * k * (z_0 - bprime * constants.g * tau * t))
		# equation (14) from Santamaria 2013
		if shifted:
			delta = 40 / self.__angular_freq	# shift anomaly in w_d
			w_d = -bprime * constants.g * tau - 2 * bprime * St \
						  * U ** 2 / c \
						  * np.exp(2 * k * (z_0 - bprime * constants.g * tau
												* (t - delta)))
		else:
			w_d = -bprime * constants.g * tau - 2 * bprime * St * U ** 2 \
						  / c * np.exp(2 * k * (z_0 - bprime * constants.g
													* tau * t))
		return u_d, w_d

	def my_analytical_drift_velocity(self, x_0=0, z_0=0, z=0, u=0, w=0, t=0):
		"""
		Computes analytical solutions for the drift velocity using my formulae.

		Parameters
		----------
		x_0 : float, default=0
			The initial horizontal position of the particle.
		z_0 : float, default=0
			The initial vertical position of the particle.
		t : float or array, default=0
			The time(s) at which to evaluate the drift velocity.

		Returns
		-------
		u_d : float or array
			The horizontal drift velocities.
		w_d : float or array
			The vertical drift velocities.

		Notes
		-----
		The formulas used to compute the analytical solutions were derived from
		my calculations following the approach from Santamaria et al., 2013.
		"""
		U = self.__max_velocity
		St = self.__stokes_num
		c = self.__phase_velocity
		bprime = 1 - self.__beta
		e_2z0t = np.exp(2 * (self.__wavenum * z_0 - St * bprime * t
											 * self.__angular_freq))
		# equations (13) and (14) derived using my equations (6) and (7)
		u_d = e_2z0t * (1 - bprime * St ** 2) * (U ** 2 / c)
		w_d = -c * St * bprime * (1 + 2 * U ** 2 / c ** 2 * e_2z0t)
		return u_d, w_d

	def analytical_period_info(self, x, z, t):
		"""
		Finds the x, z, xdot, zdot, and t values at the end of each period.

		Parameters
		----------
		x : array
			The horizontal particle positions.
		z : array
			The vertical particle positions.
		t : array
			The time over which the particle is being transported.

		Returns
		-------
		period_end : array
			The indices where the oscillatory periods end.
		current_x : array
			The horizontal particle position at the end of each period.
		current_z : array
			The vertical particle position at the end of each period.
		current_xdot : array
			The horizontal particle velocity at the end of each period.
		current_zdot : array
			The vertical particle velocity at the end of each period.
		current_t : array
			The times at which the oscillatory periods end.

		Notes
		-----
		The end points correspond to the first positive value of the horizontal
		Lagrangian velocity; for better accuracy, we should compute the point at
		which the horizontal Lagrangian velocity is 0.
		"""
		# TODO find t's where analytical xdot = 0 at x_0, z_0 = 0, 0
		# TODO save into list period_end_t
		# TODO make list of zeros called period_end_u the same length
		# TODO compute zdot's at each t, save into period_end_zdot list

		xdot, zdot = self.analytical_particle_velocity(t=t)
		period_end = []
		for i in range(1, len(xdot)):
			if xdot[i - 1] < 0 and 0 <= xdot[i]:
				period_end.append(i)

		current_x, current_z, current_xdot, current_zdot, current_t = [], [], \
																	  [], [], []
		for i in range(1, len(period_end)):
			current = period_end[i]
			current_x.append(x[current])
			current_z.append(z[current])
			current_xdot.append(xdot[current])
			current_zdot.append(zdot[current])
			current_t.append(t[current])

		return period_end, np.array(current_x), np.array(current_z), \
			   np.array(current_xdot), np.array(current_zdot), \
			   np.array(current_t)

	def analytical_averages(self, x, z, t):
		"""
		Computes the period averages of the analytical solutions for xdot, zdot.

		Parameters
		----------
		x : array
			The horizontal particle positions.
		z : array
			The vertical particle positions.
		t : array
			The time over which the particle is being transported.

		Returns
		-------
		averaged_xdot : array
			The averages of xdot over each oscillatory period.
		averaged_zdot : array
			The averages of zdot over each oscillatory period.
		period_t : array
			The times at which the oscillatory periods end.
		"""
		averaged_xdot, averaged_zdot = [], []
		period_end, _, _, _, _, period_t = self.analytical_period_info(x, z, t)
		xdot, zdot = self.analytical_particle_velocity(t=t)

		for i in range(1, len(period_end)):
			previous = period_end[i - 1]
			current = period_end[i]
			averaged_xdot.append(np.average(xdot[previous:current]))
			averaged_zdot.append(np.average(zdot[previous:current]))

		return np.array(averaged_xdot), np.array(averaged_zdot), \
			   np.array(period_t)

	def particle_trajectory(self, model, dimensional=False, x_0=0, z_0=0,
							num_periods=50, delta_t=5e-3, method='BDF'):
		"""
		Computes the position and velocity of the particle over time.

		Parameters
		----------
		model : function
			The function corresponding to the model to use to generate numerics.
		dimensional : boolean, default=False
			Whether the model is dimensional.
		x_0 : float, default=0
			The initial horizontal position of the particle.
		z_0 : float, default=0
			The initial vertical position of the particle.
		num_periods : int, default=50
			The number of oscillation periods to integrate over.
		delta_t : float, default=5e-3
			The size of the timesteps of integration.
		method : str, default='BDF'
			The method of integration to use.

		Returns
		-------
		x : array
			The horizontal positions of the particle.
		z : array
			The vertical positions of the particle.
		xdot : array
			The horizontal velocities of the particle.
		zdot : array
			The vertical velocities of the particle.
		t : array
			The times at which the model was evaluated.
		"""
		num_steps = int(np.rint(num_periods * self.__period / delta_t))
		t_span = (0, num_periods * self.__period)
		t_eval = np.linspace(0, num_periods * self.__period, num_steps)
		xdot_0, zdot_0 = self.analytical_particle_velocity(
							  dimensional=dimensional)
		sols = integrate.solve_ivp(model, t_span, [x_0, z_0, xdot_0, zdot_0],
								   method=method, t_eval=t_eval, rtol=1e-8,
								   atol=1e-10, args=(self,))
		x, z, xdot, zdot = sols.y
		t = sols.t
		return x, z, xdot, zdot, t

	def numerical_period_info(self, x, z, xdot, zdot, t):
		""" 
		Finds the x, z, xdot, zdot, and t values at the end of each period.

		Parameters
		----------
		x : array
			The horizontal particle positions.
		z : array
			The vertical particle positions.
		xdot : array
			The horizontal particle velocities.
		zdot : array
			The vertical particle velocities.
		t : array
			The times at which the model was evaluated.

		Returns
		-------
		previous_t : array
			The times corresponding to the last xdot value < 0 in each period.
		current_t : array
			The times corresponding to the first xdot value > 0 in each period.
		previous_xdot : array
			The last xdot value < 0 in each period.
		current_xdot : array
			The first xdot value > 0 in each period.
		previous_zdot : array
			The vertical velocities corresponding to the last xdot value < 0 in
			each period.
		current_zdot : array
			The vertical velocities corresponding to the first xdot value > 0 in
			each period.
		interpd_x : array
			The horizontal positions corresponding to the points where xdot = 0.
		interpd_z : array
			The vertical positions corresponding to the points where xdot = 0.
		interpd_xdot : array
			An array of zeroes, corresponding to each period endpoint.
		interpd_zdot : array
			The vertical velocities corresponding to the points where xdot = 0.
		interpd_t : array
			The times corresponding to the points where xdot = 0.
		"""
		period_end = []
		for i in range(1, len(xdot)):
			if xdot[i - 1] < 0 and 0 <= xdot[i]:
				period_end.append(i)

		previous_t, current_t, previous_xdot, current_xdot, previous_zdot, \
		current_zdot, interpd_t, interpd_xdot, interpd_zdot, interpd_x, \
		interpd_z = [], [], [], [], [], [], [], [], [], [], []

		for i in range(1, len(period_end)):
			current = period_end[i]
			previous = current - 1

			previous_t.append(t[previous])
			previous_xdot.append(xdot[previous])
			previous_zdot.append(zdot[previous])

			current_t.append(t[current])
			current_xdot.append(xdot[current])
			current_zdot.append(zdot[current])

			new_t = np.interp(0, [xdot[previous], xdot[current]],
							  [t[previous], t[current]])
			interpd_t.append(new_t)
			interpd_xdot.append(0)
			interpd_zdot.append(np.interp(new_t, [t[previous], t[current]],
							 [zdot[previous], zdot[current]]))
			interpd_x.append(np.interp(new_t, [t[previous], t[current]],
							 [x[previous], x[current]]))
			interpd_z.append(np.interp(new_t, [t[previous], t[current]],
							 [z[previous], z[current]]))

		return np.array(previous_t), np.array(current_t), \
			   np.array(previous_xdot),np.array(current_xdot), \
			   np.array(previous_zdot), np.array(current_zdot), \
			   np.array(interpd_x), np.array(interpd_z), np.array(interpd_t), \
			   np.array(interpd_xdot), np.array(interpd_zdot)


	def numerical_drift_velocity(self, x, z, xdot, zdot, t, x_0=0, z_0=0):
		"""
		Computes the numerical solutions for the drift velocity.		

		Parameters
		----------
		x : array
			The horizontal positions of the particle.
		z : array
			The vertical positions of the particle.
		xdot : array
			The horizontal velocities of the particle.
		zdot : array
			The vertical velocities of the particle.
		t : array
			The times at which the model was evaluated.
		x_0 : float, default=0
			The initial horizontal position of the particle.
		z_0 : float, default=0
			The initial vertical position of the particle.

		Returns
		-------
		u_d : array
			The numerical solutions for the horizontal drift velocity.
		w_d : array
			The numerical solutions for the vertical drift velocity.
		"""
		u_d, w_d = [], []
		_, _, _, _, _, _, interpd_x, interpd_z, interpd_t, \
			  _, _ = self.numerical_period_info(x, z, xdot, zdot, t)

		for i in range(1, len(interpd_t)):
			u_d.append((interpd_x[i] - interpd_x[i - 1]) \
							   / (interpd_t[i] - interpd_t[i - 1]))
			w_d.append((interpd_z[i] - interpd_z[i - 1]) \
							   / (interpd_t[i] - interpd_t[i - 1]))

		return np.array(u_d), np.array(w_d), np.array(interpd_t[1:])

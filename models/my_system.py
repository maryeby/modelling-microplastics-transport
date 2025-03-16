import numpy as np
from time import time
from tqdm import tqdm

from transport_framework import particle, wave, transport_system
from utils.colors import print_warning

class MyTransportSystem(transport_system.TransportSystem):
	"""Represent the transport of a particle in a linear water wave.[^1]"""

	def __init__(self, particle, flow, density_ratio):
		r"""
		Attributes
		----------
		particle : Particle (obj)
			The particle being transported.
		flow : Flow (obj)
			The flow through which the particle is transported.
		density_ratio : float
			The ratio *R* between the particle and fluid densities.
		reynolds_num : float
			The particle Reynolds number, computed as,
			$$Re_p = \frac{2\omega'A'a'}{\nu'},$$
			where *ω'*, *A'* and ν' are attributes of the wave, and *a'* is the
			radius of the particle.

		References
		----------
		[^1]: [A. Daitche (2013).](https://doi.org/10.1016/j.jcp.2013.07.024)
			  Advection of inertial particles in the presence of the history
			  force: Higher order numerical schemes.
			  *Journal of Computational Physics* 254, 93–106.
		"""
		super().__init__(particle, flow, density_ratio)
		if isinstance(flow, wave.Wave):
			self.reynolds_num = (2 * self.flow.angular_freq
								   * self.flow.amplitude
								   * np.sqrt(9 * self.particle.stokes_num 
								   / (2 * self.flow.wavenum ** 2
								   * self.flow.reynolds_num))) \
								   / self.flow.kinematic_viscosity
			if 0.1 < self.reynolds_num:
				print_warning('Particle Reynolds number' 
						   + f'(Re_p = {self.reynolds_num:.4f}) is not << 1.')

	def maxey_riley(self, t, y, include_history, hide_progress=False,
					include_H=False, order=3):
		r"""
		Evaluate the Maxey-Riley equation.

		This approach is an implementation of the integration scheme outlined in
		[1] Section 3, modified to include the buoyancy force.

		Parameters
		----------
		t : ndarray
			1D array containing `float` time series data.
		y : list
			A list of `float` data, the initial particle position and velocity.
		include_history : bool
			Whether to include history effects.
		hide_progress : bool, default=False
			Whether to hide progress output (progress bar, print statements).
		include_H : boolean, default=False
			Whether to return the values of variable H.
		order : int, default=3
			The order of the integration scheme.

		Returns
		-------
		x : ndarray
			1D array of `float` data, the horizontal particle position.
		z : ndarray
			1D array of `float` data, the vertical particle position.
		xdot : ndarray
			1D array of `float` data, the horizontal particle velocity.
		zdot : ndarray
			1D array of `float` data, the vertical particle velocity.
		t : ndarray
			1D array containing `float` time series data.
		fpg_x : ndarray
			1D array of `float` data, the horizontal fluid pressure gradient.
		fpg_z : ndarray
			1D array of `float` data, the vertical fluid pressure gradient.
		buoyancy_x : ndarray
			1D array of `float` data, the horizontal buoyancy force.
		buoyancy_z : ndarray
			1D array of `float` data, the vertical buoyancy force.
		mass_x : ndarray
			1D array of `float` data, the horizontal added mass force.
		mass_z : ndarray
			1D array of `float` data, the vertical added mass force.
		drag_x : ndarray
			1D array of `float` data, the horizontal Stokes drag.
		drag_z : ndarray
			1D array of `float` data, the vertical Stokes drag.
		H_x : ndarray
			1D array of `float` data, the horizontal H value.
		H_z : ndarray
			1D array of `float` data, the vertical H value.
		history_x : ndarray
			1D array of `float` data, the horizontal history force.
		history_z : ndarray
			1D array of `float` data, the vertical history force.

		References
		----------
		[^1]: [A. Daitche (2013).](https://doi.org/10.1016/j.jcp.2013.07.024)
			  Advection of inertial particles in the presence of the history
			  force: Higher order numerical schemes.
			  *Journal of Computational Physics* 254, 93–106.
		"""
		# initialize local variables
		R = self.density_ratio
		St = self.particle.stokes_num
		delta_t = t[1] - t[0]
		h = self.flow.wavenum * self.flow.depth if isinstance(self.flow,
			wave.Wave) else self.flow.depth

		# compute the number of time steps and create arrays to store solutions
		num_mini_steps = int(np.ceil(2 * np.sqrt(2) / delta_t))
		mini_step = 2 * delta_t / num_mini_steps
		mini_steps = np.arange(0, num_mini_steps * mini_step + mini_step,
							   mini_step)
		mini_x = np.empty((mini_steps.size, 2)) 
		mini_v = np.empty((mini_steps.size, 2))
		mini_u = np.empty((mini_steps.size, 2))
		mini_fpg = np.empty((mini_steps.size, 2))	   # fluid pressure gradient
		mini_buoyancy = np.empty((mini_steps.size, 2)) # buoyancy force
		mini_mass = np.empty((mini_steps.size, 2))	   # added mass force
		mini_drag = np.empty((mini_steps.size, 2))	   # Stokes drag
		mini_history = np.empty((mini_steps.size, 2)) if include_history else \
					   np.zeros((mini_steps.size, 2))  # history force
		num_steps = t.size - 1
		x = np.empty((t.size, 2))
		v = np.empty((t.size, 2))
		u = np.empty((t.size, 2))
		fluid_pressure_gradient = np.empty((t.size, 2))
		buoyancy = np.empty((t.size, 2))
		added_mass = np.empty((t.size, 2))
		stokes_drag = np.empty((t.size, 2))
		history = np.empty((t.size, 2)) if include_history else \
				  np.zeros((t.size, 2))

		# set initial conditions
		x[0] = y[:2]
		v[0] = y[2:]
		u[0] = self.flow.velocity(x[0, 0], x[0, 1], t[0])
		mini_x[0] = x[0]
		mini_v[0] = v[0]
		mini_u[0] = u[0]

		# immediately return if z_0 is below the depth of the water (z_0 < -h)
		if x[0, 1] <= -h:
			print('ERROR: Initial vertical position is below the seabed.')
			return x[0, 0], x[0, 1], v[0, 0], v[0, 1], t[0], \
				   0, 0, 0, 0, 0, 0, 0, 0

		# only compute alpha, beta, gamma, xi if we're including history effects
		if include_history:
			xi = np.sqrt((9 * delta_t) / (2 * np.pi)) * (R / np.sqrt(St))
			mini_xi = np.sqrt((9 * mini_step) / (2 * np.pi)) * (R / np.sqrt(St))

			# compute matrices containing the values of alpha, beta, and gamma
			if order == 1:
				mini_alpha = compute_alpha(mini_steps.size, hide_progress)
				alpha = compute_alpha(t.size, hide_progress)
			elif order == 2:
				mini_alpha = compute_alpha(2, hide_progress)
				mini_beta = compute_beta(mini_steps.size, mini_alpha[:, 1],
										 hide_progress) 
				alpha = mini_alpha
				beta = compute_beta(t.size, alpha[:, 1], hide_progress)
			else: # order == 3
				mini_alpha = compute_alpha(2, hide_progress)
				mini_beta = compute_beta(3, mini_alpha[:, 1], hide_progress) 
				mini_gamma = compute_gamma(mini_steps.size, mini_beta[:, 2],
										   hide_progress) 
				alpha = mini_alpha
				beta = mini_beta
				gamma = compute_gamma(t.size, beta[:, 2], hide_progress)

		# compute solutions for the first two intervals using finer time steps
		if not hide_progress:
			print('Computing the first two intervals using mini steps...')
		for n_prime in tqdm(range(mini_steps.size - 1),
							disable=hide_progress):
			# return immediately if the particle reaches the seabed (z < -h)
			if mini_x[n_prime, 1] <= -h:
				if not hide_progress:
					print('Simulation ended prematurely: particle reached the',
						  'seabed.')
				return mini_x[:n_prime, 0], mini_x[:n_prime, 1], \
					   mini_v[:n_prime, 0], mini_v[:n_prime, 1], \
					   mini_steps[:n_prime], \
					   mini_fpg[:n_prime, 0], mini_fpg[:n_prime, 1], \
					   mini_buoyancy[:n_prime, 0], mini_buoyancy[:n_prime, 1],\
					   mini_mass[:n_prime, 0], mini_mass[:n_prime, 1], \
					   mini_drag[:n_prime, 0], mini_drag[:n_prime, 1], \
					   mini_history[:n_prime, 0], mini_history[:n_prime, 1]

			mini_w = mini_v - mini_u
			mini_fpg = (3 / 2 * R - 1) \
						* self.flow.derivative_along_trajectory(mini_x[:, 0].T,
																mini_x[:, 1].T,
																mini_steps,
																mini_v.T).T
			mini_buoyancy[n_prime] = (1 - 3 * R / 2) * self.flow.gravity
			mini_mass = -3 / 2 * R * self.flow.dot_jacobian(mini_w.T,
															mini_x[:, 0].T,
															mini_x[:, 1].T,
															mini_steps).T
			mini_drag = -R / St * mini_w
			G = mini_fpg + mini_buoyancy[n_prime] + mini_mass + mini_drag
			sum_term = 0
			history_sum = 0

			# equation (15)
			if order == 1 or n_prime == 0:
				mini_x[n_prime + 1] = mini_x[n_prime] + mini_step \
													  * mini_v[n_prime]
				mini_u[n_prime + 1] = self.flow.velocity(mini_x[n_prime + 1, 0],
										mini_x[n_prime + 1, 1],
										mini_steps[n_prime + 1])
				if include_history:
					for j in range(n_prime + 1):
						sum_term += mini_w[n_prime - j] \
								  * (mini_alpha[j + 1, n_prime + 1] \
								  - mini_alpha[j, n_prime])
						history_sum += mini_w[n_prime - j] \
									 * mini_alpha[j, n_prime]
					mini_history[n_prime] = -mini_xi * history_sum
					mini_v[n_prime + 1] = (mini_w[n_prime] \
											+ mini_step * G[n_prime] \
											- mini_xi * sum_term) \
											/ (1 + mini_xi
											* mini_alpha[0, n_prime + 1]) \
											+ mini_u[n_prime + 1]
				else:
					mini_v[n_prime + 1] = mini_w[n_prime] \
											+ mini_step * G[n_prime] \
											+ mini_u[n_prime + 1]
			# equation (16)
			elif order == 2 or n_prime == 1:
				mini_x[n_prime + 1] = mini_x[n_prime] + mini_step / 2 \
										* (3 * mini_v[n_prime]
										- mini_v[n_prime - 1])
				mini_u[n_prime + 1] = self.flow.velocity(mini_x[n_prime + 1, 0],
										mini_x[n_prime + 1, 1],
										mini_steps[n_prime + 1])
				if include_history:
					for j in range(n_prime + 1):
						sum_term += mini_w[n_prime - j] \
								  * (mini_beta[j + 1, n_prime + 1]
								  - mini_beta[j, n_prime])
						history_sum += mini_w[n_prime - j] \
									 * mini_beta[j, n_prime]
					mini_history[n_prime] = -mini_xi * history_sum
					mini_v[n_prime + 1] = (mini_w[n_prime] + mini_step / 2 \
											* (3 * G[n_prime] - G[n_prime - 1])
											- mini_xi * sum_term) / (1 + mini_xi
											* mini_beta[0, n_prime + 1]) \
											+ mini_u[n_prime + 1]
				else:
					mini_v[n_prime + 1] = mini_w[n_prime] + mini_step / 2 \
											* (3 * G[n_prime] - G[n_prime - 1])\
											+ mini_u[n_prime + 1]
			# equation (17)
			else: # order is 3 and n_prime > 1
				mini_x[n_prime + 1] = mini_x[n_prime] + mini_step / 12 \
										* (23 * mini_v[n_prime]
										- 16 * mini_v[n_prime - 1]
										+ 5 * mini_v[n_prime - 2])
				mini_u[n_prime + 1] = self.flow.velocity(mini_x[n_prime + 1, 0],
										mini_x[n_prime + 1, 1],
										mini_steps[n_prime + 1])
				if include_history:
					for j in range(n_prime + 1):
						sum_term += mini_w[n_prime - j] \
								  * (mini_gamma[j + 1, n_prime + 1] \
								  - mini_gamma[j, n_prime])
						history_sum += mini_w[n_prime - j] \
									 * mini_gamma[j, n_prime]
					mini_history[n_prime] = -mini_xi * history_sum
					mini_v[n_prime + 1] = (mini_w[n_prime] + mini_step / 12 \
											* (23 * G[n_prime]
											- 16 * G[n_prime - 1]
											+ 5 * G[n_prime - 2]) \
											- mini_xi * sum_term) \
											/ (1 + mini_xi
											* mini_gamma[0, n_prime + 1]) \
											+ mini_u[n_prime + 1]
				else:
					mini_v[n_prime + 1] = mini_w[n_prime] + mini_step / 12 \
											* (23 * G[n_prime]
											- 16 * G[n_prime - 1]
											+ 5 * G[n_prime - 2]) \
											+ mini_u[n_prime + 1]

		# store solutions for the first two intervals
		fluid_pressure_gradient[0] = mini_fpg[0]
		buoyancy[0] = mini_buoyancy[0]
		added_mass[0] = mini_mass[0]
		stokes_drag[0] = mini_drag[0]
		history[0] = mini_history[0]
		x[1] = mini_x[int(mini_steps.size / 2)]
		v[1] = mini_v[int(mini_steps.size / 2)]
		u[1] = mini_u[int(mini_steps.size / 2)]
		fluid_pressure_gradient[1] = mini_fpg[int(mini_steps.size / 2)]
		buoyancy[1] = mini_buoyancy[int(mini_steps.size / 2)]
		added_mass[1] = mini_mass[int(mini_steps.size / 2)]
		stokes_drag[1] = mini_drag[int(mini_steps.size / 2)]
		history[1] = mini_history[int(mini_steps.size / 2)]
		x[2] = mini_x[-1]
		v[2] = mini_v[-1]
		u[2] = mini_u[-1]
		fluid_pressure_gradient[2] = mini_fpg[-1]
		buoyancy[2] = mini_buoyancy[-1]
		added_mass[2] = mini_mass[-1]
		stokes_drag[2] = mini_drag[-1]
		history[2] = mini_history[-1]

		# compute solutions for the remaining intervals
		if not hide_progress:
			print('Computing the remaining intervals...')
		for n in tqdm(range(2, num_steps), disable=hide_progress):
			# return immediately if the particle reaches the seabed (z < -h)
			if x[n, 1] <= -h:
				if not hide_progress:
					print('Simulation ended prematurely: particle reached the',
						  'seabed.')
				return x[:n, 0], x[:n, 1], v[:n, 0], v[:n, 1], t[:n], \
					   fluid_pressure_gradient[:n, 0], \
					   fluid_pressure_gradient[:n, 1], \
					   buoyancy[:n, 0], buoyancy[:n, 1], \
					   added_mass[:n, 0], added_mass[:n, 1], \
					   stokes_drag[:n, 0], stokes_drag[:n, 1], \
					   history[:n, 0], history[:n, 1]
			w = v - u
			fluid_pressure_gradient = (3 / 2 * R - 1) \
					* self.flow.derivative_along_trajectory(x[:, 0].T,
															x[:, 1].T, t, v.T).T
			buoyancy[n] = (1 - 3 * R / 2) * self.flow.gravity
			added_mass = -3 / 2 * R \
					* self.flow.dot_jacobian(w.T, x[:, 0].T, x[:, 1].T, t).T
			stokes_drag = -R / St * w
			G = fluid_pressure_gradient + buoyancy[n] + added_mass + stokes_drag
			sum_term = 0
			history_sum = 0
			if order == 1 or n == 0:
				x[n + 1] = x[n] + delta_t * v[n]
				u[n + 1] = self.flow.velocity(x[n + 1, 0], x[n + 1, 1],
											  t[n + 1])
				if include_history:
					for j in range(n + 1):
						sum_term += w[n - j] * (alpha[j + 1, n + 1]
											 - alpha[j, n])
						history_sum += w[n - j] * alpha[j, n]
					history[n] = -xi * history_sum
					v[n + 1] = (w[n] + delta_t * G[n] - xi * sum_term) \
									 / (1 + xi * alpha[0, n + 1]) + u[n + 1]
				else:
					v[n + 1] = w[n] + delta_t * G[n] + u[n + 1]
			elif order == 2 or n == 1:
				x[n + 1] = x[n] + delta_t / 2 * (3 * v[n] - v[n - 1])
				u[n + 1] = self.flow.velocity(x[n + 1, 0], x[n + 1, 1],
											  t[n + 1])
				if include_history:
					for j in range(n + 1):
						sum_term += w[n - j] * (beta[j + 1, n + 1] - beta[j, n])
						history_sum += w[n - j] * beta[j, n]
					history[n] = -xi * history_sum
					v[n + 1] = (w[n] + delta_t / 2 * (3 * G[n] - G[n - 1])
									 - xi * sum_term) \
									 / (1 + xi * beta[0, n + 1]) + u[n + 1]
				else:
					v[n + 1] = w[n] + delta_t / 2 * (3 * G[n] - G[n - 1]) \
									+ u[n + 1]
			else: # order is 3 and n > 1
				x[n + 1] = x[n] + delta_t / 12 * (23 * v[n] - 16 * v[n - 1]
								+ 5 * v[n - 2])
				u[n + 1] = self.flow.velocity(x[n + 1, 0], x[n + 1, 1],
											  t[n + 1])
				if include_history:
					for j in range(n + 1):
						sum_term += w[n - j] * (gamma[j + 1, n + 1]
											 - gamma[j, n])
						history_sum += w[n - j] * gamma[j, n]
					history[n] = -xi * history_sum
					v[n + 1] = (w[n] + delta_t / 12 * (23 * G[n] - 16 * G[n - 1]
									 + 5 * G[n - 2]) - xi * sum_term) \
									 / (1 + xi * gamma[0, n + 1]) + u[n + 1]
				else:
					v[n + 1] = w[n] + delta_t / 12 * (23 * G[n] - 16 * G[n - 1]
									+ 5 * G[n - 2]) + u[n + 1]
		H = np.copy(history)
		history[:, 0] = np.gradient(history[:, 0], t, edge_order=2)
		history[:, 1] = np.gradient(history[:, 1], t, edge_order=2)

		# truncate data after the vertical position reaches the surface
		n = np.where(0 < x[:, 1])[0]
		if 0 < len(n) and isinstance(self.flow, wave.Wave):
			n = n[0]
			print('Data truncated after particle reached the surface.')
			if include_H:
				return x[:n, 0], x[:n, 1], v[:n, 0], v[:n, 1], t[:n], \
					   fluid_pressure_gradient[:n, 0], \
					   fluid_pressure_gradient[:n, 1], \
					   buoyancy[:n, 0], buoyancy[:n, 1], \
					   added_mass[:n, 0], added_mass[:n, 1], \
					   stokes_drag[:n, 0], stokes_drag[:n, 1], \
					   H[:n, 0], H[:n, 1], history[:n, 0], history[:n, 1]
			else:
				return x[:n, 0], x[:n, 1], v[:n, 0], v[:n, 1], t[:n], \
					   fluid_pressure_gradient[:n, 0], \
					   fluid_pressure_gradient[:n, 1], \
					   buoyancy[:n, 0], buoyancy[:n, 1], \
					   added_mass[:n, 0], added_mass[:n, 1], \
					   stokes_drag[:n, 0], stokes_drag[:n, 1], \
					   history[:n, 0], history[:n, 1]

		if include_H:
			return x[:, 0], x[:, 1], v[:, 0], v[:, 1], t, \
				   fluid_pressure_gradient[:, 0], \
				   fluid_pressure_gradient[:, 1], \
				   buoyancy[:, 0], buoyancy[:, 1], added_mass[:, 0], \
				   added_mass[:, 1], stokes_drag[:, 0], stokes_drag[:, 1], \
				   H[:, 0], H[:, 1], history[:, 0], history[:, 1]
		else:
			return x[:, 0], x[:, 1], v[:, 0], v[:, 1], t, \
				   fluid_pressure_gradient[:, 0], \
				   fluid_pressure_gradient[:, 1], \
				   buoyancy[:, 0], buoyancy[:, 1], added_mass[:, 0], \
				   added_mass[:, 1], stokes_drag[:, 0], stokes_drag[:, 1], \
				   history[:, 0], history[:, 1]

def compute_alpha(size, hide_progress):
	r"""
	Create an array of the values of alpha as defined in equation (9) in [1].

	Parameters
	----------
	size : int
		The number of rows and columns for the square matrix.
	hide_progress : bool
		Whether to hide progress output (print statements).

	Returns
	-------
	ndarray
		2D square array of `float` data, the values of the coefficient alpha.

	Notes
	-----
	Alpha is computed,
	$$\alpha_j^n = \frac{4}{3} \begin{cases}
		1 & j = 0 \\
		(j - 1)^{3 / 2} + (j + 1)^{3 / 2} - 2j^{3 / 2} & 0 < j < n \\
		(n - 1)^{3 / 2} - n^{3 / 2} + \frac{3}{2} \sqrt{n} & j = n.
		\end{cases}$$
	The value of alpha may be obtained by indexing the array `arr[j, n]`.

	References
	----------
	[^1]: [A. Daitche (2013).](https://doi.org/10.1016/j.jcp.2013.07.024)
		  Advection of inertial particles in the presence of the history
		  force: Higher order numerical schemes.
		  *Journal of Computational Physics* 254, 93–106.
	"""
	if not hide_progress:
		print('Computing matrix of alpha coefficients...', end='', flush=True)
	start = time()
	arr = np.ones((size, size))
	j, n = np.indices(arr.shape, dtype='float128')

	# initialize variables for frequently used values
	coeff = np.float128(4) / np.float128(3)
	exp = np.float128(3) / np.float128(2)
	one = np.float128(1)

	# j == 0
	arr[0, 1:] = coeff

	# 0 < j < n
	mask = np.where(np.triu(arr, k=1), True, False)
	mask[0] = False
	vals = (coeff) * ((j[1:-1] - one) ** exp + (j[1:-1] + one) ** exp
				   - np.float128(2) * j[1:-1] ** exp)
	vals = vals[np.triu(vals, k=2) != 0]
	np.place(arr, mask, vals.astype('float64'))

	# j == n
	diagonal = coeff * ((n[0, 1:] - one) ** exp - n[0, 1:] ** exp
					 + exp * np.sqrt(n[0, 1:]))
	diagonal = np.insert(diagonal, 0, 0)
	np.fill_diagonal(arr, diagonal.astype('float64'))
	if not hide_progress: print('done.\t\t{:7.2f}s'.format(time() - start))
	return np.triu(arr)

def compute_beta(size, alpha, hide_progress):
	r"""
	Create an array of the values of beta as defined in [1] Section 2.

	Parameters
	----------
	size : int
		The number of rows and columns for the square matrix.
	alpha : ndarray
		2D array of `float` data, the values of the coefficient alpha at n = 1.
	hide_progress : bool
		Whether to hide progress output (print statements).

	Returns
	-------
	ndarray
		2D square array of `float` data, the values of the coefficient beta.

	References
	----------
	[^1]: [A. Daitche (2013).](https://doi.org/10.1016/j.jcp.2013.07.024)
		  Advection of inertial particles in the presence of the history
		  force: Higher order numerical schemes.
		  *Journal of Computational Physics* 254, 93–106.
	"""
	if not hide_progress:
		print('Computing matrix of beta coefficients...', end='', flush=True)
	start = time()
	arr = np.ones((size, size))
	j, n = np.indices(arr.shape, dtype='float128')
	arr[:, 0] = 0		# n = 0 (should never be called for beta)
	arr[:2, 1] = alpha	# n = 1

	# initialize variables for frequently used values
	root2 = np.sqrt(np.float128(2))
	root3 = np.sqrt(np.float128(3))

	# n = 2
	arr[0, 2] = np.float128(12) / np.float128(15) * root2
	arr[1, 2] = np.float128(16) / np.float128(15) * root2
	arr[2, 2] = np.float128(2) / np.float128(15) * root2

	if 3 < size:
		# initialize variables for frequently used values
		coeff1 = np.float128(4) / np.float128(5)
		coeff2 = np.float128(12) / np.float128(5)
		coeff3 = np.float128(8) / np.float128(15)
		exp1 = np.float128(3) / np.float128(2)
		exp2 = np.float128(5) / np.float128(2)
		one = np.float128(1)
		two = np.float128(2)
		three = np.float128(3)

		# n = 3
		arr[0, 3] = coeff1 * root2 
		arr[1, 3] = np.float128(14) / np.float128(5) * root3 - coeff2 * root2
		arr[2, 3] = -np.float128(8) / np.float128(5) * root3 + coeff2 * root2
		arr[3, 3] = coeff1 * root3 - coeff1 * root2

		# n >= 4
		arr[0, 4:] = arr[0, 3]
		arr[1, 4:] = arr[1, 3]
		arr[2, 4:] = np.float128(176) / np.float128(15) \
						- np.float128(42) / np.float128(5) * root3 \
						+ coeff2 * root2
		# j = n - 1
		mask = np.where(np.eye(size, k=1), True, False)
		mask[:, :4] = False
		vals = coeff3 * (-two * n[0, 4:] ** exp2
					  + three * (n[0, 4:] - one) ** exp2
					  - (n[0, 4:] - two) ** exp2) \
					  + two / three * (np.float128(4) * n[0, 4:] ** exp1 
					  - three * (n[0, 4:] - one) ** exp1
					  + (n[0, 4:] - two) ** exp1)
		np.place(arr, mask, vals.astype('float64'))

		# j = n
		mask = np.where(np.eye(size), True, False)
		mask[:, :4] = False
		vals = coeff3 * (n[0, 4:] ** exp2 - (n[0, 4:] - one) ** exp2) \
					  + two / three * (-three * n[0, 4:] ** exp1
					  + (n[0, 4:] - one) ** exp1) + two * np.sqrt(n[0, 4:])
		np.place(arr, mask, vals.astype('float64'))

		# 2 < j < n - 1
		mask = np.where(np.triu(arr) == 1, True, False)
		vals = coeff3 * ((j[3:-2, 5:] + two) ** exp2
					  - three * (j[3:-2, 5:] + one) ** exp2
					  + three * j[3:-2, 5:] ** exp2
					  - (j[3:-2, 5:] - one) ** exp2) \
					  + two / three * (-(j[3:-2, 5:] + two) ** exp1
					  + three * (j[3:-2, 5:] + one) ** exp1
					  - three * j[3:-2, 5:] ** exp1
					  + (j[3:-2, 5:] - one) ** exp1)
		vals = vals[np.triu(vals) != 0]
		np.place(arr, mask, vals.astype('float64'))
	if not hide_progress: print('done.\t\t{:7.2f}s'.format(time() - start))
	return np.triu(arr)

def compute_gamma(size, beta, hide_progress):
	r"""
	Create an array of the values of gamma as defined in [1] Section 2.

	Parameters
	----------
	size : int
		The number of rows and columns for the square matrix.
	beta : ndarray
		2D array of `float` data, the values of the coefficient beta at n = 1.
	hide_progress : bool
		Whether to hide progress output (print statements).

	Returns
	-------
	ndarray
		2D square array of `float` data, the values of the coefficient gamma.

	References
	----------
	[^1]: [A. Daitche (2013).](https://doi.org/10.1016/j.jcp.2013.07.024)
		  Advection of inertial particles in the presence of the history
		  force: Higher order numerical schemes.
		  *Journal of Computational Physics* 254, 93–106.
	"""
	if not hide_progress:
		print('Computing matrix of gamma coefficients...', end='', flush=True)
	start = time()
	arr = np.ones((size, size))
	j, n = np.indices(arr.shape)
	arr[:, :2] = 0		# n = 0 and n = 1 (should never be called for gamma)
	arr[:3, 2] = beta	# n = 2
	
	# initialize variables for frequently used values
	coeff = np.float128(16) / np.float128(105)
	exp1 = np.float128(3) / np.float128(2)
	exp2 = np.float128(5) / np.float128(2)
	exp3 = np.float128(7) / np.float128(2)
	one = np.float128(1)
	two = np.float128(2)
	three = np.float128(3)
	four = np.float128(4)
	five = np.float128(5)
	six = np.float128(6)
	eight = np.float128(8)
	nine = np.float128(9)
	root2 = np.sqrt(two)
	root3 = np.sqrt(three)
	root5 = np.sqrt(five)
	root6 = np.sqrt(six)

	# n = 3
	arr[0, 3] = np.float128(68) / np.float128(105) * root3
	arr[1, 3] = np.float128(6) / np.float128(7) * root3
	arr[2, 3] = np.float128(12) / np.float128(35) * root3
	arr[3, 3] = np.float128(16) / np.float128(105) * root3

	# n = 4
	arr[0, 4] = np.float128(244) / np.float128(315) * root2
	arr[1, 4] = np.float128(1888) / np.float128(315) \
				- np.float128(976) / np.float128(315) * root2
	arr[2, 4] = -np.float128(656) / np.float128(105) \
				+ np.float128(488) / np.float128(105) * root2
	arr[3, 4] = np.float128(544) / np.float128(105) \
				- np.float128(976) / np.float128(315) * root2
	arr[4, 4] = -np.float128(292) / np.float128(315) \
				+ np.float128(244) / np.float128(315) * root2

	# n = 5
	arr[0, 5] = arr[0, 4]
	arr[1, 5] = np.float128(362) / np.float128(105) * root3 \
				- np.float128(976) / np.float128(315) * root2
	arr[2, 5] = np.float128(500) / np.float128(63) * root5 \
				- np.float128(1448) / np.float128(105) * root3 \
				+ np.float128(488) / np.float128(105) * root2
	arr[3, 5] = -np.float128(290) / np.float128(21) * root5 \
				+ np.float128(724) / np.float128(35) * root3 \
				- np.float128(976) / np.float128(315) * root2
	arr[4, 5] = np.float128(220) / np.float128(21) * root5 \
				- np.float128(1448) / np.float128(105) * root3 \
				+ np.float128(244) / np.float128(315) * root2
	arr[5, 5] = -np.float128(164) / np.float128(63) * root5 \
				+ np.float128(362) / np.float128(105) * root3

	# n = 6
	arr[0, 6] = arr[0, 4]
	arr[1, 6] = arr[1, 5]
	arr[2, 6] = np.float128(5584) / np.float128(315) \
				- np.float128(1448) / np.float128(105) * root3 \
				+ np.float128(488) / np.float128(105) * root2
	arr[3, 6] = np.float128(344) / np.float128(21) * root6 \
				- np.float128(22336) / np.float128(315) \
				+ np.float128(724) / np.float128(35) * root3 \
				- np.float128(976) / np.float128(315) * root2
	arr[4, 6] = -np.float128(1188) / np.float128(35) * root6 \
				+ np.float128(11168) / np.float128(105) \
				- np.float128(1448) / np.float128(105) * root3 \
				+ np.float128(244) / np.float128(315) * root2
	arr[5, 6] = np.float128(936) / np.float128(35) * root6 \
				- np.float128(22336) / np.float128(315) \
				+ np.float128(362) / np.float128(105) * root3
	arr[6, 6] = -np.float128(754) / np.float128(105) * root6 \
				+ np.float128(5584) / np.float128(315)

	# n >= 7
	arr[0, 7:] = arr[0, 4]
	arr[1, 7:] = arr[1, 5]
	arr[2, 7:] = arr[2, 6]
	arr[3, 7:] = np.float128(1130) / np.float128(63) * root5 \
					- np.float128(22336) / np.float128(315) \
					+ np.float128(724) / np.float128(35) * root3 \
					- np.float128(976) / np.float128(315) * root2

	# j = n - 3
	mask = np.where(np.eye(size, k=3), True, False)
	mask[:, :7] = False
	vals = coeff * (n[0, 7:] ** exp3 - four * (n[0, 7:] - two) ** exp3
				 + six * (n[0, 7:] - three) ** exp3
				 - four * (n[0, 7:] - four) ** exp3
			  	 + (n[0, 7:] - five) ** exp3) \
				 - eight / np.float128(15) * n[0, 7:] ** exp2 \
				 + four / nine * n[0, 7:] ** exp1 \
				 + eight / nine * (n[0, 7:] - two) ** exp1 \
				 - four / three * (n[0, 7:] - three) ** exp1 \
				 + eight / nine * (n[0, 7:] - four) ** exp1 \
				 - two / nine * (n[0, 7:] - five) ** exp1
	np.place(arr, mask, vals.astype('float64'))

	# j = n - 2
	mask = np.where(np.eye(size, k=2), True, False)
	mask[:, :7] = False
	vals = coeff * ((n[0, 7:] - four) ** exp3 
				 - four * (n[0, 7:] - three) ** exp3
				 + six * (n[0, 7:] - two) ** exp3 - three * n[0, 7:] ** exp3) \
				 + np.float128(32) / np.float128(15) * n[0, 7:] ** exp2 \
				 - two * n[0, 7:] ** exp1 \
				 - four / three * (n[0, 7:] - two) ** exp1 \
			  	 + eight / nine * (n[0, 7:] - three) ** exp1 \
				 - two / nine * (n[0, 7:] - four) ** exp1
	np.place(arr, mask, vals.astype('float64'))

	# j = n - 1
	mask = np.where(np.eye(size, k=1), True, False)
	mask[:, :7] = False
	vals = coeff * (three * n[0, 7:] ** exp3 - four * (n[0, 7:] - two) ** exp3
				 + (n[0, 7:] - three) ** exp3) \
				 - eight / three * n[0, 7:] ** exp2 \
				 + four * n[0, 7:] ** exp1 \
				 + eight / nine * (n[0, 7:] - two) ** exp1 \
			  	 - two / nine * (n[0, 7:] - three) ** exp1
	np.place(arr, mask, vals.astype('float64'))

	# j = n
	mask = np.where(np.eye(size), True, False)
	mask[:, :7] = False
	vals = coeff * ((n[0, 7:] - two) ** exp3 - n[0, 7:] ** exp3) \
				 + np.float128(16) / np.float128(15) * n[0, 7:] ** exp2 \
				 - np.float128(22) / nine * n[0, 7:] ** exp1 \
				 - two / nine * (n[0, 7:] - two) ** exp1 \
				 + two * np.sqrt(n[0, 7:])
	np.place(arr, mask, vals.astype('float64'))

	# 3 < j < n - 3
	mask = np.where(np.triu(arr) == 1, True, False)
	vals = coeff * ((j[4:-4, 8:] + two) ** exp3 + (j[4:-4, 8:] - two) ** exp3
				 - four * (j[4:-4, 8:] + one) ** exp3
				 - four * (j[4:-4, 8:] - one) ** exp3
				 + six * j[4:-4, 8:] ** exp3) \
				 + two / nine * (four * (j[4:-4, 8:] + one) ** exp1
				 + four * (j[4:-4, 8:] - one) ** exp1
				 - (j[4:-4, 8:] + two) ** exp1
				 - (j[4:-4, 8:] - two) ** exp1 - six * j[4:-4, 8:] ** exp1)
	vals = vals[np.triu(vals) != 0]
	np.place(arr, mask, vals.astype('float64'))
	if not hide_progress: print('done.\t\t{:7.2f}s'.format(time() - start))
	return np.triu(arr)

def compute_drift_velocity(x, z, xdot, t):
	r"""
	Compute the Stokes drift velocity numerically.

	Parameters
	----------
	x : ndarray
		1D array of `float` data, the horizontal particle position.
	z : ndarray
		1D array of `float` data, the vertical particle position.
	xdot : ndarray
		1D array of `float` data, the horizontal particle velocity.
	t : ndarray
		1D array containing `float` time series data.

	Returns
	-------
	x_crossings, z_crossings : ndarray
		The horizontal and vertical particle position at the end of each period.
	u_d : ndarray
		1D array of `float` data, the horizontal Stokes drift velocities.
	w_d : ndarray
		1D array of `float` data, the vertical Stokes drift velocities.
	t : ndarray
		1D array containing `float` time series data for the end of each period.

	Notes
	-----
	The drift velocity $$\bar{\mathbf{u}} = \langle \bar{u}, \bar{w} \rangle$$
	is computed using the distance travelled by the particle averaged over each
	wave period *p*,
	$$\bar{\mathbf{u}} = \frac{\mathbf{x}_{p + 1} - \mathbf{x}_p}
	{t_{p + 1} - t_p}.$$
	"""
	# find the estimated endpoints of the periods
	estimated_endpoints = []
	for i in range(1, len(xdot)):
		if xdot[i - 1] < 0 and 0 <= xdot[i]:
			estimated_endpoints.append(i)

	# find the exact endpoints of the periods using interpolation
	interpd_x, interpd_z, interpd_t = [], [], []
	for i in range(len(estimated_endpoints)):
		current = estimated_endpoints[i]
		previous = current - 1

		new_t = np.interp(0, [xdot[previous], xdot[current]], [t[previous],
															   t[current]])
		interpd_t.append(new_t)
		interpd_x.append(np.interp(new_t, [t[previous], t[current]],
								   [x[previous], x[current]]))
		interpd_z.append(np.interp(new_t, [t[previous], t[current]],
								   [z[previous], z[current]]))

	# compute drift velocity
	u_bar, w_bar = [], []
	for i in range(1, len(interpd_t)):
		u_bar.append((interpd_x[i] - interpd_x[i - 1])
				 / (interpd_t[i] - interpd_t[i - 1]))
		w_bar.append((interpd_z[i] - interpd_z[i - 1])
				 / (interpd_t[i] - interpd_t[i - 1]))

	# return results
	x_crossings = np.array(interpd_x)
	z_crossings = np.array(interpd_z)
	u_bar = np.array(u_bar)
	w_bar = np.array(w_bar)
	t = np.array(interpd_t)
	return x_crossings, z_crossings, u_bar, w_bar, t

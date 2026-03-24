import numpy as np
import scipy.integrate as integrate
from time import time
from tqdm import tqdm

from models import quiescent_flow
from transport_framework import particle, transport_system

class RelaxingTransportSystem(transport_system.TransportSystem):
	"""Represent the transport of a particle in a quiescent flow.[^1]"""

	def __init__(self, particle, flow, density_ratio):
		r"""
		Attributes
		----------
		particle : Particle (obj)
			The particle being transported.
		flow : Flow (obj)
			The flow through which the particle is transported.
		density_ratio : float
			The ratio $\beta$ between the particle and fluid densities[^1],
			$$\beta = \frac{\rho'_p}{\rho'_f}.$$
		stokes_num : float
			The density-dependent Stokes number,
			$$St = \beta \widehat{St}.$$
		alpha : float
			A relationship between the density ratio and Stokes number[^1],
			$$\alpha = \frac{2}{3R \widehat{St}},$$ with
			$$R = \frac{1 + 2\beta}{3},$$
			used to compute the asymptotic behavior of the system.
		gamma : float
			A relationship between the density ratio and Stokes number[^1],
			$$\gamma = \frac{1}{R} \sqrt{\frac{2}{\widehat{St}}},$$
			used to compute the asymptotic behavior of the system.

		References
		----------
		[^1]: [S. G. Prasath et al. (2019)](
			  https://doi.org/10.1017/jfm.2019.194)
			  Accurate solution method for the Maxey–Riley equation, and the
			  effects of Basset history. *Journal of Fluid Mechanics*
			  868, 428–460.
		[^2]: [A. Daitche (2013).](https://doi.org/10.1016/j.jcp.2013.07.024)
			  Advection of inertial particles in the presence of the history
			  force: Higher order numerical schemes. *Journal of Computational
			  Physics* 254, 93–106.
		"""
		super().__init__(particle, flow, density_ratio)
		r = (1 + 2 * density_ratio) / 3
		self.alpha = 2 / (3 * r * particle.stokes_hat)
		self.gamma = 1 / r * np.sqrt(2 / particle.stokes_hat)

	def set_stokes_num(self):
		r"""Set the density-dependent Stokes number $St$."""
		self.stokes_num = self.particle.stokes_hat * self.density_ratio

	def asymptotic_velocity(self, t):
		r"""
		Compute the leading order asymptotic behavior of the particle velocity.

		Parameters
		----------
		t : float or ndarray
			Float or 1D array containing `float` time series data.

		Returns
		-------
		float or ndarray
			The asymptotic particle velocity.

		Notes
		-----
		The computation is based on eq (4.7) from Ref. 1,
		$$q^{(2)}(0, t) \approx c(\alpha, \gamma)
			- \frac{\sigma \gamma}{\alpha^2 \sqrt{\pi t}}
			+ \mathcal{O}(t^{3 / 2}),$$
		with a sign change on the singular term.
		"""
		return 1 / (np.sqrt(np.pi) * t ** (3 / 2)) * (self.gamma \
				 / (2 * self.alpha ** 2))

	def maxey_riley(self, t, y):
		r"""
		Evaluate the Maxey-Riley equation without history effects.
		
		Parameters
		----------
		t : ndarray
			1D array containing `float` time series data.
		y : list
			A list of `float` data, the initial particle position and velocity.

		Returns
		-------
		ndarray
			1D array of `float` data, the particle velocity and acceleration.

		Notes
		-----
		The Maxey-Riley equation is expressed as,
		$$\frac{\mathrm{d}\boldsymbol{x}}{\mathrm{d}t} = \boldsymbol{v},$$
		$$\frac{\mathrm{d}\boldsymbol{v}}{\mathrm{d}t} = \frac{1}{R}
		  \frac{\mathrm{d}\boldsymbol{u}}{\mathrm{d}t}
			+ \Bigg(1 - \frac{1}{R}\Bigg) \boldsymbol{g}
			+ \alpha(\boldsymbol{u} - \boldsymbol{v}).$$
		"""
		# initialize local variables and update the particle and fluid histories
		r = (1 + 2 * self.density_ratio) / 3
		x, z = y[:2]
		particle_velocity = y[2:]
		fluid_velocity = self.flow.velocity(x, z, t)

		# compute terms on the RHS of the M-R equation
		stokes_drag = (fluid_velocity - particle_velocity) * self.alpha
		fluid_pressure_gradient = 1 / r * self.flow.material_derivative(x, z, t)

		# M-R equation
		particle_acceleration = stokes_drag + fluid_pressure_gradient
		return np.concatenate((particle_velocity, particle_acceleration))

	def full_maxey_riley(self, t, y, order):
		r"""
		Evaluate the Maxey-Riley equation with history effects from [2].

		Parameters
		----------
		t : ndarray
			1D array containing `float` time series data.
		y : list
			A list of `float` data, the initial particle position and velocity.
		order : int
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
		"""
		# initialize local variables
		r = (1 + 2 * self.density_ratio) / 3
		sthat = self.particle.stokes_hat
		delta_t = t[1] - t[0]
		xi = 2 / r * np.sqrt(delta_t) * np.sqrt(1 / (2 * np.pi * sthat))

		# compute the number of time steps and create arrays to store x, v data
		mini_steps = 2 * int(np.sqrt(2) / delta_t)
		mini_step = delta_t / (mini_steps / 2)
		mini_xi = 2 / r * np.sqrt(mini_step) * np.sqrt(1 / (2 * np.pi * sthat))
		mini_x = np.empty((mini_steps + 1, 2))
		mini_v = np.empty((mini_steps + 1, 2))
		num_steps = t.size - 1
		x = np.empty((t.size, 2))
		v = np.empty((t.size, 2))
		x[0] = y[:2]	# initial particle position
		v[0] = y[2:]	# initial particle velocity
		mini_x[0] = x[0]
		mini_v[0] = v[0]

		# compute matrices containing the values of alpha, beta, and gamma
		if order == 1:
			mini_alpha = compute_alpha(mini_steps + 1)
			alpha = compute_alpha(t.size)
		elif order == 2:
			mini_alpha = compute_alpha(2)
			mini_beta = compute_beta(mini_steps + 1, mini_alpha[:, 1])
			alpha = mini_alpha
			beta = compute_beta(t.size, alpha[:, 1])
		else: # order == 3
			mini_alpha = compute_alpha(2)
			mini_beta = compute_beta(3, mini_alpha[:, 1])
			mini_gamma = compute_gamma(mini_steps + 1, mini_beta[:, 2])
			alpha = mini_alpha
			beta = mini_beta
			gamma = compute_gamma(t.size, beta[:, 2])

		# compute solutions for the first two intervals using finer time steps
		for n_prime in range(mini_steps):
			g = -2 / (3 * r * sthat) * mini_v
			sum_term = 0
			if order == 1 or n_prime == 0:
				for j in range(n_prime + 1):
					sum_term += mini_v[n_prime - j] \
							  * (mini_alpha[j + 1, n_prime + 1] \
							  - mini_alpha[j, n_prime])
				mini_x[n_prime + 1] = x[n_prime] + mini_step * mini_v[n_prime]
				mini_v[n_prime + 1] = (v[n_prime] + mini_step * g[n_prime]
												  - mini_xi * sum_term) \
												  / (1 + mini_xi
												  * mini_alpha[0, n_prime + 1])
			elif order == 2 or n_prime == 1:
				for j in range(n_prime + 1):
					sum_term += mini_v[n_prime - j] \
							  * (mini_beta[j + 1, n_prime + 1] 
							  - mini_beta[j, n_prime])
				mini_x[n_prime + 1] = mini_x[n_prime] + mini_step / 2 \
										* (3 * mini_v[n_prime]
										- mini_v[n_prime - 1])
				mini_v[n_prime + 1] = (mini_v[n_prime] + mini_step / 2 \
										* (3 * g[n_prime] - g[n_prime - 1]) \
								 		- mini_xi * sum_term) \
										/ (1 + mini_xi * mini_beta[0,
																   n_prime + 1])
			else: # order is 3 and n_prime > 1
				for j in range(n_prime + 1):
					sum_term += mini_v[n_prime - j] \
							  * (mini_gamma[j + 1, n_prime + 1]
							  - mini_gamma[j, n_prime])
				mini_x[n_prime + 1] = mini_x[n_prime] + mini_step / 12 \
										* (23 * mini_v[n_prime]
										- 16 * mini_v[n_prime - 1]
										+ 5 * mini_v[n_prime - 2])
				mini_v[n_prime + 1] = (mini_v[n_prime] + mini_step / 12 \
										* (23 * g[n_prime] - 16 * g[n_prime - 1]
								 		+ 5 * g[n_prime - 2]) \
										- mini_xi * sum_term) \
								 		/ (1 + mini_xi * gamma[0, n_prime + 1])

		# store solutions for the first two intervals
		x[1] = mini_x[int(mini_steps / 2)]
		v[1] = mini_v[int(mini_steps / 2)]
		x[2] = mini_x[-1]
		v[2] = mini_v[-1]
		
		# compute solutions for the remaining intervals
		for n in tqdm(range(2, num_steps)):
			g = -2 / (3 * r * sthat) * v
			sum_term = 0
			if order == 1 or n == 0:
				for j in range(n + 1):
					sum_term += v[n - j] * (alpha[j + 1, n + 1] - alpha[j, n])
				x[n + 1] = x[n] + delta_t * v[n]
				v[n + 1] = (v[n] + delta_t * g[n] - xi * sum_term) \
								 / (1 + xi * alpha[0, n + 1])
			elif order == 2 or n == 1:
				for j in range(n + 1):
					sum_term += v[n - j] * (beta[j + 1, n + 1] - beta[j, n])
				x[n + 1] = x[n] + delta_t / 2 * (3 * v[n] - v[n - 1])
				v[n + 1] = (v[n] + delta_t / 2 * (3 * g[n] - g[n - 1])
								 - xi * sum_term) / (1 + xi * beta[0, n + 1])
			else: # order is 3 and n > 1
				for j in range(n + 1):
					sum_term += v[n - j] * (gamma[j + 1, n + 1] - gamma[j, n])
				x[n + 1] = x[n] + delta_t / 12 * (23 * v[n] - 16 * v[n - 1]
								+ 5 * v[n - 2])
				v[n + 1] = (v[n] + delta_t / 12 * (23 * g[n] - 16 * g[n - 1]
								 + 5 * g[n - 2]) - xi * sum_term) \
								 / (1 + xi * gamma[0, n + 1])
		return x[:, 0], x[:, 1], v[:, 0], v[:, 1], t

	def run_numerics(self, x_0, z_0, xdot_0, zdot_0, num_periods, delta_t,
					 include_history, order=3):
		"""
		Compute the position and velocity of the particle over time.

		Parameters
		----------
		x_0, z_0 : float
			The initial horizontal and vertical position of the particle.
		xdot_0, zdot_0 : float
			The initial horizontal and vertical velocity of the particle.
		num_periods : int
			The number of periods to integrate over.
		delta_t : float
			The size of the time steps used for integration.
		include_history : bool
			Whether to include history effects.
		order : int, default=3
			The order of the integration scheme (first, second, or third).

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
		"""
		# initialize parameters for the solver
		t_final = num_periods * self.flow.period
		t_span = (0, t_final)
		t_eval = np.arange(0, t_final, delta_t)
		y = [x_0, z_0, xdot_0, zdot_0]

		# run computations
		if include_history:	
			x, z, xdot, zdot, t = self.full_maxey_riley(t_eval, y, order=order)
		else:
			sols = integrate.solve_ivp(self.maxey_riley, t_span, y,
									   method='BDF', t_eval=t_eval,
									   rtol=1e-8, atol=1e-10)
			# unpack solutions
			x, z, xdot, zdot = sols.y
			t = sols.t
		return x, z, xdot, zdot, t

def compute_alpha(size):
	r"""
	Create an array of the values of alpha as defined in equation (9) from [2].

	Parameters
	----------
	size : int
		The number of rows and columns for the square matrix.

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
	"""
	print('Computing matrix of alpha coefficients...', end='', flush=True)
	start = time()
	arr = np.ones((size, size))
	j, n = np.indices(arr.shape)
	arr[0, 1:] = 4 / 3 # j == 0

	# 0 < j < n
	mask = np.where(np.triu(arr, k=1), True, False)
	mask[0] = False
	vals = (4 / 3) * ((j[1:-1] - 1) ** (3 / 2) + (j[1:-1] + 1) ** (3 / 2)
			  - 2 * j[1:-1] ** (3 / 2))
	vals = vals[np.triu(vals, k=2) != 0]
	np.place(arr, mask, vals)

	# j == n
	diagonal = 4 / 3 * ((n[0, 1:] - 1) ** (3 / 2) - n[0, 1:] ** (3 / 2)
				 + (3 / 2) * np.sqrt(n[0, 1:]))
	diagonal = np.insert(diagonal, 0, 0)
	np.fill_diagonal(arr, diagonal)
	finish = time()
	print('done.\t\t{:5.2f}s'.format(finish - start))
	return np.triu(arr)

def compute_beta(size, alpha):
	r"""
	Create an array of the values of beta as defined in [2] Section 2.

	Parameters
	----------
	size : int
		The number of rows and columns for the square matrix.
	alpha : ndarray
		2D array of `float` data, the values of the coefficient alpha at n = 1.

	Returns
	-------
	ndarray
		2D square array of `float` data, the values of the coefficient beta.
	"""
	print('Computing matrix of beta coefficients...', end='', flush=True)
	start = time()
	arr = np.ones((size, size))
	j, n = np.indices(arr.shape)
	arr[:, 0] = 0				# n = 0 (should never be called for beta)
	arr[:alpha.size, 1] = alpha	# n = 1

	# n = 2
	arr[0, 2] = 12 / 15 * np.sqrt(2)
	arr[1, 2] = 16 / 15 * np.sqrt(2)
	arr[2, 2] = 2 / 15 * np.sqrt(2)

	if 3 < size:
		# n = 3
		arr[0, 3] = 4 / 5 * np.sqrt(2) 
		arr[1, 3] = 14 / 5 * np.sqrt(3) - 12 / 5 * np.sqrt(2)
		arr[2, 3] = -(8 / 5) * np.sqrt(3) + 12 / 5 * np.sqrt(2)
		arr[3, 3] = 4 / 5 * np.sqrt(3) - 4 / 5 * np.sqrt(2)

		# n >= 4
		arr[0, 4:] = 4 / 5 * np.sqrt(2)
		arr[1, 4:] = 14 / 5 * np.sqrt(3) - 12 / 5 * np.sqrt(2)
		arr[2, 4:] = 176 / 15 - 42 / 5 * np.sqrt(3) + 12 / 5 * np.sqrt(2)
	
		# j = n - 1
		mask = np.where(np.eye(size, k=1), True, False)
		mask[:, :4] = False
		vals = 8 / 15 * (-2 * n[0, 4:] ** (5 / 2)
				 + 3 * (n[0, 4:] - 1) ** (5 / 2) - (n[0, 4:] - 2) ** (5 / 2)) \
				 + 2 / 3 * (4 * n[0, 4:] ** (3 / 2)
				 - 3 * (n[0, 4:] - 1) ** (3 / 2) + (n[0, 4:] - 2) ** (3 / 2))
		np.place(arr, mask, vals)

		# j = n
		mask = np.where(np.eye(size), True, False)
		mask[:, :4] = False
		vals = 8 / 15 * (n[0, 4:] ** (5 / 2) - (n[0, 4:] - 1) ** (5 / 2)) \
				 + 2 / 3 * (-3 * n[0, 4:] ** (3 / 2) 
				 + (n[0, 4:] - 1) ** (3 / 2)) + 2 * np.sqrt(n[0, 4:])
		np.place(arr, mask, vals)

		# 2 < j < n - 1
		mask = np.where(np.triu(arr) == 1, True, False)
		vals = 8 / 15 * ((j[3:-2, 5:] + 2) ** (5 / 2) \
				 - 3 * (j[3:-2, 5:] + 1) ** (5 / 2)
				 + 3 * j[3:-2, 5:] ** (5 / 2) - (j[3:-2, 5:] - 1) ** (5 / 2)) \
				 + 2 / 3 * (-(j[3:-2, 5:] + 2) ** (3 / 2) \
				 + 3 * (j[3:-2, 5:] + 1) ** (3 / 2) \
				 - 3 * j[3:-2, 5:] ** (3 / 2) + (j[3:-2, 5:] - 1) ** (3 / 2))
		vals = vals[np.triu(vals) != 0]
		np.place(arr, mask, vals)
	finish = time()
	print('done.\t\t{:5.2f}s'.format(finish - start))
	return np.triu(arr)

def compute_gamma(size, beta):
	r"""
	Create an array of the values of gamma as defined in [2] Section 2.

	Parameters
	----------
	size : int
		The number of rows and columns for the square matrix.
	beta : ndarray
		2D array of `float` data, the values of the coefficient beta at n = 1.

	Returns
	-------
	ndarray
		2D square array of `float` data, the values of the coefficient gamma.
	"""
	print('Computing matrix of gamma coefficients...', end='', flush=True)
	start = time()
	arr = np.ones((size, size))
	j, n = np.indices(arr.shape)
	arr[:, :2] = 0		# n = 0 and n = 1 (should never be called for gamma)
	arr[:3, 2] = beta	# n = 2
	
	# n = 3
	arr[0][3] = 68 / 105 * np.sqrt(3)
	arr[1][3] = 6 / 7 * np.sqrt(3)
	arr[2][3] = 12 / 35 * np.sqrt(3)
	arr[3][3] = 16 / 105 * np.sqrt(3)

	# n = 4
	arr[0][4] = 244 / 315 * np.sqrt(2)
	arr[1][4] = 1888 / 315 - 976 / 315 * np.sqrt(2)
	arr[2][4] = -(656 / 105) + 488 / 105 * np.sqrt(2)
	arr[3][4] = 544 / 105 - 976 / 315 * np.sqrt(2)
	arr[4][4] = -(292 / 315) + 244 / 315 * np.sqrt(2)

	# n = 5
	arr[0][5] = 244 / 315 * np.sqrt(2)
	arr[1][5] = 362 / 105 * np.sqrt(3) - 976 / 315 * np.sqrt(2)
	arr[2][5] = 500 / 63 * np.sqrt(5) - 1448 / 105 * np.sqrt(3) + 488 / 105 \
					* np.sqrt(2)
	arr[3][5] = -(290 / 21) * np.sqrt(5) + 724 / 35 * np.sqrt(3) - 976 / 315 \
							* np.sqrt(2)
	arr[4][5] = 220 / 21 * np.sqrt(5) - 1448 / 105 * np.sqrt(3) + 244 / 315 \
					* np.sqrt(2)
	arr[5][5] = -(164 / 63) * np.sqrt(5) + 362 / 105 * np.sqrt(3)

	# n = 6
	arr[0][6] = 244 / 315 * np.sqrt(2)
	arr[1][6] = 362 / 105 * np.sqrt(3) - 976 / 315 * np.sqrt(2)
	arr[2][6] = 5584 / 315 - 1448 / 105 * np.sqrt(3) + 488 / 105 * np.sqrt(2)
	arr[3][6] = 344 / 21 * np.sqrt(6) - 22336 / 315 + 724 / 35 * np.sqrt(3) \
					- 976 / 315 * np.sqrt(2)
	arr[4][6] = -(1188 / 35) * np.sqrt(6) + 11168 / 105 - 1448 / 105 \
							 * np.sqrt(3) + 244 / 315 * np.sqrt(2)
	arr[5][6] = 936 / 35 * np.sqrt(6) - 22336 / 315 + 362 / 105 * np.sqrt(3)
	arr[6][6] = -(754 / 105) * np.sqrt(6) + 5584 / 315

	# n >= 7
	arr[0][7:] = 244 / 315 * np.sqrt(2)
	arr[1][7:] = 362 / 105 * np.sqrt(3) - 976 / 315 * np.sqrt(2)
	arr[2][7:] = 5584 / 315 - 1448 / 105 * np.sqrt(3) + 488 / 105 * np.sqrt(2)
	arr[3][7:] = 1130 / 63 * np.sqrt(5) - 22336 / 315 + 724 / 35 * np.sqrt(3) \
					  - 976 / 315 * np.sqrt(2)

	# j = n - 3
	mask = np.where(np.eye(size, k=3), True, False)
	mask[:, :7] = False
	vals = 16 / 105 * (n[0, 7:] ** (7 / 2) - 4 * (n[0, 7:] - 2) ** (7 / 2)
			  + 6 * (n[0, 7:] - 3) ** (7 / 2) - 4 * (n[0, 7:] - 4) ** (7 / 2)
			  + (n[0, 7:] - 5) ** (7 / 2)) - 8 / 15 * n[0, 7:] ** (5 / 2) \
			  + 4 / 9 * n[0, 7:] ** (3 / 2) + 8 / 9 * (n[0, 7:] - 2) ** (3 / 2)\
			  - 4 / 3 * (n[0, 7:] - 3) ** (3 / 2) \
			  + 8 / 9 * (n[0, 7:] - 4) ** (3 / 2) \
			  - 2 / 9 * (n[0, 7:] - 5) ** (3 / 2)
	np.place(arr, mask, vals)

	# j = n - 2
	mask = np.where(np.eye(size, k=2), True, False)
	mask[:, :7] = False
	vals = 16 / 105 * ((n[0, 7:] - 4) ** (7 / 2) - 4 * (n[0, 7:] - 3) ** (7 / 2)
			  + 6 * (n[0, 7:] - 2) ** (7 / 2) - 3 * n[0, 7:] ** (7 / 2)) \
			  + 32 / 15 * n[0, 7:] ** (5 / 2) - 2 * n[0, 7:] ** (3 / 2) \
			  - 4 / 3 * (n[0, 7:] - 2) ** (3 / 2) \
			  + 8 / 9 * (n[0, 7:] - 3) ** (3 / 2) \
			  - 2 / 9 * (n[0, 7:] - 4) ** (3 / 2)
	np.place(arr, mask, vals)

	# j = n - 1
	mask = np.where(np.eye(size, k=1), True, False)
	mask[:, :7] = False
	vals = 16 / 105 * (3 * n[0, 7:] ** (7 / 2) - 4 * (n[0, 7:] - 2) ** (7 / 2)
			  + (n[0, 7:] - 3) ** (7 / 2)) - 8 / 3 * n[0, 7:] ** (5 / 2) \
			  + 4 * n[0, 7:] ** (3 / 2) + 8 / 9 * (n[0, 7:] - 2) ** (3 / 2) \
			  - 2 / 9 * (n[0, 7:] - 3) ** (3 / 2)
	np.place(arr, mask, vals)

	# j = n
	mask = np.where(np.eye(size), True, False)
	mask[:, :7] = False
	vals = 16 / 105 * ((n[0, 7:] - 2) ** (7 / 2) - n[0, 7:] ** (7 / 2)) \
			  + 16 / 15 * n[0, 7:] ** (5 / 2) - 22 / 9 * n[0, 7:] ** (3 / 2) \
			  - 2 / 9 * (n[0, 7:] - 2) ** (3 / 2) + 2 * np.sqrt(n[0, 7:])
	np.place(arr, mask, vals)

	# 3 < j < n - 3
	mask = np.where(np.triu(arr) == 1, True, False)
	vals = 16 / 105 * ((j[4:-4, 8:] + 2) ** (7 / 2) \
			  + (j[4:-4, 8:] - 2) ** (7 / 2)
			  - 4 * (j[4:-4, 8:] + 1) ** (7 / 2) 
			  - 4 * (j[4:-4, 8:] - 1) ** (7 / 2) 
			  + 6 * j[4:-4, 8:] ** (7 / 2)) \
			  + 2 / 9 * (4 * (j[4:-4, 8:] + 1) ** (3 / 2)
			  + 4 * (j[4:-4, 8:] - 1) ** (3 / 2) - (j[4:-4, 8:] + 2) ** (3 / 2)
			  - (j[4:-4, 8:] - 2) ** (3 / 2) - 6 * j[4:-4, 8:] ** (3 / 2))
	vals = vals[np.triu(vals) != 0]
	np.place(arr, mask, vals)
	finish = time()
	print('done.\t\t{:5.2f}s'.format(finish - start))
	return np.triu(arr)

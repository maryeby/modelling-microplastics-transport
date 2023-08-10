import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
import scipy.integrate as integrate
from time import time
from tqdm import tqdm
from models import rotating_flow
from transport_framework import particle, transport_system

class RotatingTransportSystem(transport_system.TransportSystem):
	""" 
	Represents the transport of a relaxing particle in a quiescent fluid flow.
	"""

	def __init__(self, particle, flow, density_ratio):
		r"""
		Attributes
		----------
		particle : Particle (obj)
			The particle being transported.
		flow : Flow (obj)
			The flow through which the particle is transported.
		density_ratio : float
			The ratio between the particle's density and the fluid's density.
		epsilon : float
			A relationship between the Stokes number and density ratio,
			$$\epsilon = \frac{St}{R}.$$
		sigma : float
			A parameter related to the density ratio,
			$$\sigma = \Bigg(1 - \frac{3}{2} R \Bigg) \textbf{g},$$
			used to compute the asymptotic behavior of the system.
		alpha : float
			A relationship between the density ratio and Stokes number,
			$$\alpha = \frac{R}{St},$$
			used to compute the asymptotic behavior of the system.
		gamma : float
			Another relationship between the density ratio and Stokes number,
			$$\gamma = \frac{3}{2} R \sqrt{\frac{2}{St}},$$
			used to compute the asymptotic behavior of the system.
		"""
		super().__init__(particle, flow, density_ratio)
		self.epsilon = self.particle.stokes_num / self.density_ratio
		self.alpha = self.density_ratio / self.particle.stokes_num
		self.gamma = 3 / 2 * self.density_ratio \
					   * np.sqrt(2 / self.particle.stokes_num)

	def asymptotic_velocity(self, t):
		r"""
		Computes the leading order asymptotic behavior of the particle velocity
		based on eq (4.7) from Prasath et al. (2019),
		$$q^{(2)}(0, t) \approx c(\alpha, \gamma)
			- \frac{\sigma \gamma}{\alpha^2 \sqrt{\pi t}}
			+ \mathcal{O}(t^{3 / 2}),$$
		with a sign change on the singular term.

		Parameters
		----------
		t : float or array
			The time(s) to use in the computations.

		Returns
		-------
		float or array
			The asymptotic particle velocity.
		"""
		return 1 / (np.sqrt(np.pi) * t ** (3 / 2)) * (self.gamma \
				 / (2 * self.alpha ** 2))

	def derivative_along_trajectory(self, xdot, zdot):
		r"""
		Computes the derivative of the fluid along the trajectory of the
		particle,
		$$\frac{\mathrm{d}\textbf{u}}{\mathrm{d}t}
			= \frac{\partial \textbf{u}}{\partial t}
			+ \textbf{v} \cdot \nabla \textbf{u}.$$

		Parameters
		----------
		xdot : float or array
			The value(s) of the horizontal velocity of the particle.
		zdot : float
			The value(s) of the vertical velocity of the particle.

		Returns
		-------
		Array
			The horizontal and vertical components of the solution.
		"""
		return np.transpose(np.array([-zdot, xdot]))

	def maxey_riley(self, t, y):
		r"""
		Evaluates the Maxey-Riley equation,
		$$\frac{\mathrm{d}\textbf{x}}{\mathrm{d}t} = \textbf{v},$$
		$$\frac{\mathrm{d}\textbf{v}}{\mathrm{d}t} = \frac{\textbf{u}
			- \textbf{v}}{\epsilon}
			+ \frac{3R}{2} \frac{\mathrm{d}\textbf{u}}{\mathrm{d}t}
			+ (1 - \frac{3R}{2}) \textbf{g}
			- \sqrt{\frac{9}{2\pi}} \frac{R}{\sqrt{St}} \int_0^t
			\frac{1}{\sqrt{t - s}} \mathrm{d}s
			[\textbf{v} - \textbf{u}] \frac{\mathrm{d}}{\mathrm{d}s},$$
		with $$R = \frac{2\rho_f}{\rho_f + 2\rho_p}, \quad Re = \frac{UL}{\nu},
			\quad St = \frac{2}{9} \Bigg(\frac{a}{L}\Bigg)^2 Re.$$
		
		Parameters
		----------
		t : float
			The time to use in the computations.
		y : list (array-like)
			A list containing the x, z, xdot, zdot values to use.

		Returns
		-------
		Array
			The components of the particle's velocity and acceleration.
		"""
		# initialize local variables and update the particle and fluid histories
		R = self.density_ratio
		x, z = y[:2]
		particle_velocity = y[2:]
		fluid_velocity = self.flow.velocity(x, z)

		# compute terms on the RHS of the M-R equation
		stokes_drag = (fluid_velocity - particle_velocity) / self.epsilon
		fluid_pressure_gradient = 3 * R / 2 \
									* self.flow.material_derivative(x, z, t)

		# M-R equation
		particle_acceleration = stokes_drag + fluid_pressure_gradient
		return np.concatenate((particle_velocity, particle_acceleration))

	def full_maxey_riley(self, t, y, order):
		r"""
		Implements the integration scheme for the full Maxey-Riley equation with
		history effects, as outlined in Daitche (2013) Section 3.

		Parameters
		----------
		t : array
			The times when the Maxey-Riley equation should be evaluated.
		y : list (array-like)
			A list containing the initial particle position and velocity.
		order : int
			The order of the integration scheme.

		Returns
		-------
		Array
			The components of the particle's position and velocity.
		"""
		# initialize local variables
		R = self.density_ratio
		St = self.particle.stokes_num
		delta_t = t[1] - t[0]
		xi = np.sqrt((9 * delta_t) / (2 * np.pi)) * (R / np.sqrt(St))

		# compute the number of time steps and create arrays to store solutions
		mini_steps = 2 * int(np.sqrt(2) / delta_t)
		mini_step = delta_t / (mini_steps / 2)
		mini_xi = np.sqrt((9 * mini_step) / (2 * np.pi)) * (R / np.sqrt(St))
		mini_x = np.empty((mini_steps + 1, 2)) 
		mini_v = np.empty((mini_steps + 1, 2))
		mini_u = np.empty((mini_steps + 1, 2))
		num_steps = t.size - 1
		x = np.empty((t.size, 2))
		v = np.empty((t.size, 2))
		u = np.empty((t.size, 2))

		# set initial conditions
		x[0] = y[:2]
		v[0] = y[2:]
		u[0] = self.flow.velocity(x[0, 0], x[0, 1])
		mini_x[0] = x[0]
		mini_v[0] = v[0]
		mini_u[0] = u[0]

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
			mini_w = mini_v - mini_u
			G = (3 / 2 * R - 1) \
				* self.derivative_along_trajectory(mini_v[:, 0], mini_v[:, 1]) \
				- 3 / 2 * R \
				* np.transpose(np.array([-mini_w[:, 1], mini_w[:, 0]])) \
				- R / St * mini_w
			sum_term = 0
			if order == 1 or n_prime == 0:
				for j in range(n_prime + 1):
					sum_term += mini_w[n_prime - j] \
							  * (mini_alpha[j + 1, n_prime + 1] \
							  - mini_alpha[j, n_prime])
				mini_x[n_prime + 1] = mini_x[n_prime] + mini_step \
													  * mini_v[n_prime]
				mini_u[n_prime + 1] = self.flow.velocity(mini_x[n_prime + 1, 0],
														 mini_x[n_prime + 1, 1])
				mini_v[n_prime + 1] = (mini_w[n_prime] + mini_step * G[n_prime]\
										- mini_xi * sum_term) \
								 		/ (1 + mini_xi
										* mini_alpha[0, n_prime + 1]) \
										+ mini_u[n_prime + 1]
			elif order == 2 or n_prime == 1:
				for j in range(n_prime + 1):
					sum_term += mini_w[n_prime - j] \
							  * (mini_beta[j + 1, n_prime + 1]
							  - mini_beta[j, n_prime])
				mini_x[n_prime + 1] = mini_x[n_prime] + mini_step / 2 \
										* (3 * mini_v[n_prime]
										- mini_v[n_prime - 1])
				mini_u[n_prime + 1] = self.flow.velocity(mini_x[n_prime + 1, 0],
														 mini_x[n_prime + 1, 1])
				mini_v[n_prime + 1] = (mini_w[n_prime] + mini_step / 2 \
										* (3 * G[n_prime] - G[n_prime - 1])
								 		- mini_xi * sum_term) / (1 + mini_xi
										* mini_beta[0, n_prime + 1]) \
								 		+ mini_u[n_prime + 1]
			else: # order is 3 and n_prime > 1
				for j in range(n_prime + 1):
					sum_term += mini_w[n_prime - j] \
							  * (mini_gamma[j + 1, n_prime + 1] \
							  - mini_gamma[j, n_prime])
				mini_x[n_prime + 1] = mini_x[n_prime] + mini_step / 12 \
										* (23 * mini_v[n_prime]
										- 16 * mini_v[n_prime - 1]
										+ 5 * mini_v[n_prime - 2])
				mini_u[n_prime + 1] = self.flow.velocity(mini_x[n_prime + 1, 0],
														 mini_x[n_prime + 1, 1])
				mini_v[n_prime + 1] = (mini_w[n_prime] + mini_step / 12 \
										* (23 * G[n_prime] - 16 * G[n_prime - 1]
								 		+ 5 * G[n_prime - 2]) \
										- mini_xi * sum_term) \
								 		/ (1 + mini_xi
										* mini_gamma[0, n_prime + 1]) \
										+ mini_u[n_prime + 1]

		# store solutions for the first two intervals
		x[1] = mini_x[int(mini_steps / 2)]
		v[1] = mini_v[int(mini_steps / 2)]
		u[1] = mini_u[int(mini_steps / 2)]
		x[2] = mini_x[-1]
		v[2] = mini_v[-1]
		u[2] = mini_u[-1]

		# compute solutions for the remaining intervals
		for n in tqdm(range(2, num_steps)):
			w = v - u
			G = (3 / 2 * R - 1) \
				   * self.derivative_along_trajectory(v[:, 0], v[:, 1]) \
				   - 3 / 2 * R * np.transpose(np.array([-w[:, 1], w[:, 0]])) \
				   - R / St * w
			sum_term = 0
			if order == 1 or n == 0:
				for j in range(n + 1):
					sum_term += w[n - j] * (alpha[j + 1, n + 1] - alpha[j, n])
				x[n + 1] = x[n] + delta_t * v[n]
				u[n + 1] = self.flow.velocity(x[n + 1, 0], x[n + 1, 1])
				v[n + 1] = (w[n] + delta_t * G[n] - xi * sum_term) \
								 / (1 + xi * alpha[0, n + 1]) + u[n + 1]
			elif order == 2 or n == 1:
				for j in range(n + 1):
					sum_term += w[n - j] * (beta[j + 1, n + 1] - beta[j, n])
				x[n + 1] = x[n] + delta_t / 2 * (3 * v[n] - v[n - 1])
				u[n + 1] = self.flow.velocity(x[n + 1, 0], x[n + 1, 1])
				v[n + 1] = (w[n] + delta_t / 2 * (3 * G[n] - G[n - 1])
								 - xi * sum_term) / (1 + xi * beta[0, n + 1]) \
								 + u[n + 1]
			else: # order is 3 and n > 1
				for j in range(n + 1):
					sum_term += w[n - j] * (gamma[j + 1, n + 1] - gamma[j, n])
				x[n + 1] = x[n] + delta_t / 12 * (23 * v[n] - 16 * v[n - 1]
								+ 5 * v[n - 2])
				u[n + 1] = self.flow.velocity(x[n + 1, 0], x[n + 1, 1])
				v[n + 1] = (w[n] + delta_t / 12 * (23 * G[n] - 16 * G[n - 1]
								 + 5 * G[n - 2]) - xi * sum_term) \
								 / (1 + xi * gamma[0, n + 1]) + u[n + 1]
		return x[:, 0], x[:, 1], v[:, 0], v[:, 1], t

	def run_numerics(self, include_history, order=3, x_0=0, z_0=0, xdot_0=1,
					 zdot_0=1, num_periods=50, delta_t=5e-3, method='BDF'):
		"""
		Computes the position and velocity of the particle over time.

		Parameters
		----------
		include_history : boolean
			Whether to include history effects.
		order : int
			The order of the integration scheme (first, second, or third).
		x_0 : float, default=0
			The initial horizontal position of the particle.
		z_0 : float, default=0
			The initial vertical position of the particle.
		xdot_0 : float, default=1
			The initial horizontal velocity of the particle.
		zdot_0 : float, default=1
			The initial vertical velocity of the particle.
		num_periods : int, default=50
			The number of oscillation periods to integrate over.
		delta_t : float, default=5e-3
			The size of the time steps used for integration.
		method : str, default='BDF'
			The method of integration to use when neglecting history effects.

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
		# initialize parameters for the solver
		t_final = num_periods * self.flow.period
		t_span = (0, t_final)
		t_eval = np.arange(0, t_final + delta_t, delta_t)
		y = [x_0, z_0, xdot_0, zdot_0]

		# run computations
		if include_history:	
			x, z, xdot, zdot, t = self.full_maxey_riley(t_eval, y, order=order)
		else:
			sols = integrate.solve_ivp(self.maxey_riley, t_span, y,
									   method=method, t_eval=t_eval,
									   rtol=1e-8, atol=1e-10)
			# unpack solutions
			x, z, xdot, zdot = sols.y
			t = sols.t
		return x, z, xdot, zdot, t

def compute_alpha(size):
	r"""
	Computes a matrix containing the values of alpha as defined in equation (9)
	from Daitche (2013),
	$$\alpha_j^n = \frac{4}{3} \begin{cases}
		1 & j = 0 \\
		(j - 1)^{3 / 2} + (j + 1)^{3 / 2} - 2j^{3 / 2} & 0 < j < n \\
		(n - 1)^{3 / 2} - n^{3 / 2} + \frac{3}{2} \sqrt{n} & j = n,
		\end{cases}$$
	so the value of alpha may be obtained by indexing the array as `arr[j, n]`.

	Parameters
	----------
	size : int
		The number of rows and columns for the square matrix.

	Returns
	-------
	Array
		The matrix containing the values of the coefficient alpha.
	"""
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
	finish = time()
	print('done.\t\t{:7.2f}s'.format(finish - start))
	return np.triu(arr)

def compute_beta(size, alpha):
	r"""
	Computes a matrix containing the values of beta as defined in Section 2 of
	Daitche (2013). The value $$\beta_j^n$$
	may be obtained by indexing the array as `arr[j, n]`.

	Parameters
	----------
	size : int
		The number of rows and columns for the square matrix.
	alpha : array-like
		The values of the coefficient alpha at n = 1.

	Returns
	-------
	Array
		The matrix containing the values of the coefficient beta.
	"""
	print('Computing matrix of beta coefficients...', end='', flush=True)
	start = time()
	arr = np.ones((size, size))
	j, n = np.indices(arr.shape, dtype='float128')
	arr[:, 0] = 0		# n = 0 (should never be called for beta)
	arr[:2, 1] = alpha	# n = 1

	root2 = np.sqrt(np.float128(2))
	root3 = np.sqrt(np.float128(3))

	# n = 2
	arr[0, 2] = np.float128(12) / np.float128(15) * root2
	arr[1, 2] = np.float128(16) / np.float128(15) * root2
	arr[2, 2] = np.float128(2) / np.float128(15) * root2

	if 3 < size:
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
	finish = time()
	print('done.\t\t{:7.2f}s'.format(finish - start))
	return np.triu(arr)

def compute_gamma(size, beta):
	r"""
	Computes a matrix containing the values of gamma as defined in Section 2 of
	Daitche (2013). The value $$\gamma_j^n$$
	may be obtained by indexing the array as `arr[j, n]`.

	Parameters
	----------
	size : int
		The number of rows and columns for the square matrix.
	beta : array-like
		The values of the coefficient beta at n = 2.

	Returns
	-------
	Array
		The matrix containing the values of the coefficient gamma.
	"""
	print('Computing matrix of gamma coefficients...', end='', flush=True)
	start = time()
	arr = np.ones((size, size))
	j, n = np.indices(arr.shape)
	arr[:, :2] = 0		# n = 0 and n = 1 (should never be called for gamma)
	arr[:3, 2] = beta	# n = 2
	
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
	finish = time()
	print('done.\t\t{:7.2f}s'.format(finish - start))
	return np.triu(arr)

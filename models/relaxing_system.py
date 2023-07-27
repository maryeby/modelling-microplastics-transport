import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import time
import numpy as np
import scipy.integrate as integrate
from tqdm import tqdm
from models import quiescent_flow
from transport_framework import particle, transport_system

class RelaxingTransportSystem(transport_system.TransportSystem):
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
		fluid_velocity = self.flow.velocity(x, z, t)

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
		# initialize local variables and set initial conditions
		R = self.density_ratio
		St = self.particle.stokes_num
		delta_t = t[1] - t[0]
		xi = np.sqrt((9 * delta_t) / (2 * np.pi)) * (R / np.sqrt(St))
		num_steps = t.size - 1
		x = np.empty((t.size, 2))
		v = np.empty((t.size, 2))
		x[0] = y[:2]
		v[0] = y[2:]
		alpha = compute_alpha(t.size)
		if order == 2:
			beta = compute_beta(t.size, alpha[:, 1])
		elif order == 3:
			beta = compute_beta(t.size, alpha[:, 1])
			gamma = compute_gamma(t.size, beta[:, 2])

		for n in tqdm(range(num_steps)):
			G = -R / St * v
			sum_term = 0
			if order == 1 or n == 0:
				for j in range(n + 1):
					sum_term += v[n - j] * (alpha[j + 1, n + 1] - alpha[j, n])
				x[n + 1] = x[n] + delta_t * v[n]
				v[n + 1] = (v[n] + delta_t * G[n] - xi * sum_term) \
								 / (1 + xi * alpha[0, n + 1])
			elif order == 2 or n == 1:
				for j in range(n + 1):
					sum_term += v[n - j] * (beta[j + 1, n + 1] - beta[j, n])
				x[n + 1] = x[n] + delta_t / 2 * (3 * v[n] - v[n - 1])
				v[n + 1] = (v[n] + delta_t / 2 * (3 * G[n] - G[n - 1])
								 - xi * sum_term) / (1 + xi * beta[0, n + 1])
			else: # order is 3 and n > 1
				for j in range(n + 1):
					sum_term += v[n - j] * (gamma[j + 1, n + 1] - gamma[j, n])
				x[n + 1] = x[n] + delta_t / 12 * (23 * v[n] - 16 * v[n - 1]
								+ 5 * v[n - 2])
				v[n + 1] = (v[n] + delta_t / 12 * (23 * G[n] - 16 * G[n - 1]
								 + 5 * G[n - 2]) - xi * sum_term) \
								 / (1 + xi * gamma[0, n + 1])
		return x[:, 0], x[:, 1], v[:, 0], v[:, 1], t

	def run_numerics(self, include_history, order=3,
					 x_0=0, z_0=0, xdot_0=1, zdot_0=1, num_periods=50,
					 delta_t=5e-3, method='BDF'):
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
		t_eval = np.arange(0, t_final, delta_t)
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
	print('Computing matrix of alpha coefficients...', end='')
	start = time.time()
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
	finish = time.time()
	print('done.\t\t{:.2f}s'.format(finish - start))
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
	print('Computing matrix of beta coefficients...', end='')
	start = time.time()
	arr = np.ones((size, size))
	j, n = np.indices(arr.shape)
	arr[:, 0] = 0 	  # n = 0 (should never be called for beta)
	arr[:, 1] = alpha # n = 1

	# n = 2
	arr[0, 2] = 12 / 15 * np.sqrt(2)
	arr[1, 2] = 16 / 15 * np.sqrt(2)
	arr[2, 2] = 2 / 15 * np.sqrt(2)

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
	vals = 8 / 15 * (-2 * n[0, 4:] ** (5 / 2) + 3 * (n[0, 4:] - 1) ** (5 / 2)
			 - (n[0, 4:] - 2) ** (5 / 2)) + 2 / 3 * (4 * n[0, 4:] ** (3 / 2)
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
	finish = time.time()
	print('done.\t\t{:.2f}s'.format(finish - start))
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
	print('Computing matrix of gamma coefficients...', end='')
	start = time.time()
	arr = np.ones((size, size))
	j, n = np.indices(arr.shape)
	arr[:, :2] = 0		# n = 0 and n = 1 (should never be called for gamma)
	arr[:, 2] = beta	# n = 2
	
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
	finish = time.time()
	print('done.\t\t{:.2f}s'.format(finish - start))
	return np.triu(arr)

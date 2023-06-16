import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
import scipy.integrate as integrate
from models import quiescent_flow
from transport_framework import particle, transport_system

class SedimentingTransportSystem(transport_system.TransportSystem):
	""" 
	Represents the transport of a sedimenting particle in a quiescent flow.
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
		terminal_velocity : float
			The asymptotic behavior of the particle velocity **v** at long
			times, based on eq (4.8) from Prasath (2019),
			$$c(\alpha, gamma) = \frac{\sigma}{\alpha}.$$
		"""
		super().__init__(particle, flow, density_ratio)
		self.epsilon = self.particle.stokes_num / self.density_ratio
		self.sigma = (1 - 3 / 2 * self.density_ratio) * self.flow.gravity[1]
		self.alpha = self.density_ratio / self.particle.stokes_num
		self.gamma = 3 / 2 * self.density_ratio \
					   * np.sqrt(2 / self.particle.stokes_num)
		self.terminal_velocity = self.sigma / self.alpha

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
		return self.terminal_velocity - self.sigma * self.gamma \
									  / (self.alpha ** 2 * np.sqrt(np.pi * t))

	def maxey_riley(self, t, y, delta_t, t_final, include_history):
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
		delta_t : float
			The size of the time steps used for integration.
		t_final : float
			The final time of integration.

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
		history = 0

		if include_history:
			self.update_history(fluid_velocity, particle_velocity)
			history = self.compute_history(delta_t, t)

		# compute non-history terms on the RHS of the M-R equation
		stokes_drag = (fluid_velocity - particle_velocity) / self.epsilon
		buoyancy_force = (1 - 3 * R / 2) * self.flow.gravity
		fluid_pressure_gradient = 3 * R / 2 \
									* self.flow.material_derivative(x, z, t)

		# M-R equation
		particle_acceleration = stokes_drag + buoyancy_force \
											+ fluid_pressure_gradient \
									 		+ history
		print('{:.0%}'.format(t / t_final), end='\r')
		return np.concatenate((particle_velocity, particle_acceleration))

	def update_history(self, fluid_velocity, particle_velocity):
		"""
		Updates the history of the fluid velocity **u** and particle velocity
		**v**.

		Parameters
		----------
		fluid_velocity : array
			The components of the fluid velocity **u**.
		particle_velocity : array
			The components of the particle velocity **v**.
		"""
		self.flow.history.append(fluid_velocity)
		self.particle.history.append(particle_velocity)

	def compute_history(self, delta_t, t):
		r"""
		Computes the Basset-Boussinesq history term,
		$$-\sqrt{\frac{9}{2\pi}} \frac{R}{\sqrt{St}} \int_0^t
			\frac{1}{\sqrt{t - s}} \mathrm{d}s [\textbf{v} - \textbf{u}]
			\frac{\mathrm{d}}{\mathrm{d}s},$$
		using the quadrature scheme,
		$$\int_0^t \frac{1}{\sqrt{t - s}} \mathrm{d}s [\textbf{v} - \textbf{u}]
			\frac{\mathrm{d}}{\mathrm{d}s} \approx \sqrt{\Delta t}
			\sum_{j=0}^{n}
			\alpha_j^n f(s_{n-j}),$$
		with
		$$f(s) = \textbf{v} - \textbf{u},$$
		*n* steps, and
		$$\alpha_j^n = \frac{4}{3} \begin{cases}
			1 & j = 0 \\
			(j - 1)^{3 / 2} + (j + 1)^{3 / 2} - 2j^{3 / 2} & 0 < j < n \\
			(n - 1)^{3 / 2} - n^{3 / 2} + \frac{3}{2} \sqrt{n} & j = n.
			\end{cases}$$

		Parameters
		----------
		delta_t : float
			The size of the time steps used for integration.
		t : float
			The time at which to evaluate the history term.

		Returns
		-------
		Array
			The components of the history term.
		"""
		# initialize local variables
		integrand = np.empty(2,)
		coefficient = -np.sqrt(9 / (2 * np.pi)) * self.density_ratio \
					  / np.sqrt(self.particle.stokes_num)
		f_s = np.array(self.particle.history) - np.array(self.flow.history)
		num_steps = int(t / delta_t)
		alpha = 4 / 3

		# compute the sum
		for j in range(num_steps + 1):
			if j == 0:
				alpha *= 1
			elif j == num_steps:
				alpha *= (num_steps - 1) ** (3 / 2) - num_steps ** (3 / 2) \
									+ (3 / 2) * np.sqrt(num_steps)
			else: # 0 < j < num_steps
				alpha *= (j - 1) ** (3 / 2) + (j + 1) ** (3 / 2) \
											- 2 * j ** (3 / 2)
			integrand += alpha * f_s[num_steps - j]
		integrand *= np.sqrt(delta_t)
		return coefficient * integrand

	def run_numerics(self, include_history, x_0=0, z_0=0, xdot_0=1, zdot_0=1,
					 num_periods=50, delta_t=5e-3, method='BDF'):
		"""
		Computes the position and velocity of the particle over time.

		Parameters
		----------
		include_history : boolean
			Whether to include history effects.
		x_0 : float, default=0
			The initial horizontal position of the particle.
		z_0 : float, default=0
			The initial vertical position of the particle.
		num_periods : int, default=50
			The number of oscillation periods to integrate over.
		delta_t : float, default=5e-3
			The size of the time steps used for integration.
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
		# initialize parameters for the solver
		num_steps = int(np.rint(num_periods * self.flow.period / delta_t))
		t_final = num_periods * self.flow.period
		t_span = (0, t_final)	
		t_eval = np.linspace(0, t_final, num_steps)
		args=(delta_t, t_final, include_history)

		# run computations
		sols = integrate.solve_ivp(self.maxey_riley, t_span,
								   [x_0, z_0, xdot_0, zdot_0],
								   method=method, t_eval=t_eval,
								   rtol=1e-8, atol=1e-10, args=args)
		# unpack and return solutions
		x, z, xdot, zdot = sols.y
		t = sols.t
		return x, z, xdot, zdot, t

from abc import ABC, abstractmethod
import numpy as np
import scipy.integrate as integrate

class TransportSystem:
	"""
	Represents the transport of an inertial particle in a fluid flow.
	"""

	def __init__(self, particle, flow, density_ratio):
		"""
		Attributes
		----------
		particle : Particle (obj)
			The particle being transported.
		flow : Flow (obj)
			The flow through which the particle is transported.
		gravity : float
			The gravity _**g**_.
		density_ratio : float
			The ratio between the particle's density and the fluid's density.
		"""
		self.particle = particle
		self.flow = flow
		self.density_ratio = density_ratio

	@abstractmethod
	def maxey_riley(self, t, y):
		"""
		Evaluates the Maxey-Riley equation without the history term.
		
		Parameters
		----------
		t : float
			The time to use in the computations
		y : list (array-like)
			A list containing the x, z, xdot, zdot values to use.

		Returns
		-------
		Array
			The components of the particle's velocity and acceleration.
		"""
		pass

	def run_numerics(self, equation, order=2, x_0=0, z_0=0, num_periods=50,
					 delta_t=5e-3, method='BDF'):
		"""
		Computes the position and velocity of the particle over time.

		Parameters
		----------
		equation : fun
			The equation to evaluate, either M-R or the inertial equation.
		order : int
			The order at which to evaluate the inertial equation.
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

		Notes
		-----
		The velocity of the particle is set to the initial velocity of the
		fluid.
		"""
		# initial parameters
		num_steps = int(np.rint(num_periods * self.flow.period / delta_t))
		t_span = (0, num_periods * self.flow.period)
		t_eval = np.linspace(0, num_periods * self.flow.period, num_steps)
		xdot_0, zdot_0 = self.flow.velocity(x_0, z_0, 0)

		# run computations
		if 'maxey_riley' in str(equation):
			sols = integrate.solve_ivp(equation, t_span,
									   [x_0, z_0, xdot_0, zdot_0],
									   method=method, t_eval=t_eval,
									   rtol=1e-8, atol=1e-10)
		elif 'inertial' in str(equation):
			sols = integrate.solve_ivp(equation, t_span,
									   [x_0, z_0, xdot_0, zdot_0],
									   method=method, t_eval=t_eval,
									   rtol=1e-8, atol=1e-10, args=(order,))
		else:
			print('Could not recognize equation.')

		# unpack and return solutions
		x, z, xdot, zdot = sols.y
		t = sols.t
		return x, z, xdot, zdot, t

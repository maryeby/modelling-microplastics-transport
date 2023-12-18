from abc import ABC, abstractmethod
import numpy as np

class TransportSystem:
	"""
	Represents the transport of an inertial particle in a fluid flow.
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
			The ratio between the particle and fluid densities.
		"""
		self.particle = particle
		self.flow = flow
		self.density_ratio = density_ratio

	@abstractmethod
	def maxey_riley(self, t, y):
		"""
		Evaluates the Maxey-Riley equation.
		
		Parameters
		----------
		t : array
			The time(s) to use in the computations.
		y : list (array-like)
			A list containing the initial particle position and velocity.

		Returns
		-------
		Array
			The components of the particle's position and velocity, and the
			times where the Maxey-Riley equation was evaluated.
		"""
		pass

	@abstractmethod
	def run_numerics(self, x_0, z_0, num_periods, delta_t, method='BDF'):
		"""
		Computes the position and velocity of the particle over time.

		Parameters
		----------
		x_0 : float
			The initial horizontal position of the particle.
		z_0 : float
			The initial vertical position of the particle.
		num_periods : int
			The number of periods to integrate over.
		delta_t : float
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
		pass

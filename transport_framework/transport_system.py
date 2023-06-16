from abc import ABC, abstractmethod
import numpy as np

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

	@abstractmethod
	def run_numerics(self, x_0=0, z_0=0, num_periods=50, delta_t=5e-3,
					 method='BDF'):
		"""
		Computes the position and velocity of the particle over time.

		Parameters
		----------
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
		pass

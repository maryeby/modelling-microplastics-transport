from abc import ABC, abstractmethod
import numpy as np

class TransportSystem:
	"""Represent the transport of an inertial particle in a fluid flow."""

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
		Evaluate the Maxey-Riley equation.
		
		Parameters
		----------
		t : ndarray
			1D array containing `float` time series data.
		y : list
			A list of `float` data, the initial particle position and velocity.

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
		pass

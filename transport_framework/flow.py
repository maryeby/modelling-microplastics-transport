from abc import ABC, abstractmethod
import numpy as np

class Flow(ABC):
	"""Represents a fluid flow."""

	def __init__(self, depth):
		r"""
		Attributes
		----------
		depth : float
			The depth of the fluid *h*.
		history : list (array-like)
			The history of the velocity of the flow **u**.
		period : float
			A parameter used in the computation of the timespan over which to
			integrate.
		"""
		self.depth = depth
		self.history = []
		self.period = 1

	@abstractmethod
	def velocity(self, x, z, t):
		"""
		Computes the fluid velocity _**u**_.

		Parameters
		----------
		x : float or array
			The x position(s) at which to evaluate the velocity.
		z : float or array
			The z position(s) at which to evaluate the velocity.
		t : float or array
			The time(s) at which to evaluate the velocity.

		Returns
		-------
		Array containing the velocity field vector components _u_ and _w_.
		"""
		pass

	@abstractmethod
	def material_derivative(self, x, z, t): 
		"""
		Computes the Lagrangian derivative.

		Parameters
		----------
		x : float or array
			The x position(s) at which to evaluate the fluid velocity.
		z : float or array
			The z position(s) at which to evaluate the velocity and derivative.
		t : float or array
			The time(s) at which to evaluate the velocity.

		Returns
		-------
		Array containing the material derivative vector components.
		"""
		pass

	@abstractmethod
	def material_derivative2(self, x, z, t):
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

		Returns
		-------
		Array containing the second order material derivative vector components.
		"""
		pass

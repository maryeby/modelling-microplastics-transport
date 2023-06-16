import numpy as np
import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
from transport_framework import flow

class CouetteFlow(flow.Flow):
	"""Represents a quiescent fluid flow."""

	def __init__(self, depth=15):
		r"""
		Attributes
		----------
		depth : float, default=15
			The depth of the fluid *h*.
		history : list (array-like)
			The history of the velocity of the flow **u**.
		period : float
			A parameter used in the computation of the timespan over which to
			integrate.
		lambda_ : float
			The constant lambda used in the velocity profile.
		gravity : float
			The gravity _**g**_.
		"""
		super().__init__(depth)
		self.lambda_ = 1
		self.set_gravity()
	
	def set_gravity(self):
		"""Defines the gravity _**g**_."""
		self.gravity = np.array([0, -1])
	
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
		return np.array([0, z * self.lambda_])

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
		u, w = self.velocity(x, z, t)
		return np.array([0, u * self.lambda_])

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
		u, w = self.velocity(x, z, t)
		return np.array([0, w * self.lambda_ ** 2])

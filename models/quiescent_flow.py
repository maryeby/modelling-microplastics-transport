import numpy as np
import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
from transport_framework import flow

class QuiescentFlow(flow.Flow):
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
		gravity : float
			The gravity _**g**_.
		"""
		super().__init__(depth)
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
		return np.array([0, 0])

	def partial_t(self, x, z, t):
		r"""
		Computes the partial derivative of the fluid **u** = (_u_, _w_) with
		respect to time.

		Parameters
		----------
		x : float or array
			The x position(s) at which to evaluate the derivative.
		z : float or array
			The z position(s) at which to evaluate the derivative.
		t : float or array
			The time(s) at which to evaluate the derivative.

		Returns
		-------
		Array containing the partial time derivative vector components.
		"""
		return np.array([0, 0])

	def partial_x(self, x, z, t):
		r"""
		Computes the partial derivative of the fluid **u** = (_u_, _w_) with
		respect to the horizontal position _x_.

		Parameters
		----------
		x : float or array
			The x position(s) at which to evaluate the derivative.
		z : float or array
			The z position(s) at which to evaluate the derivative.
		t : float or array
			The time(s) at which to evaluate the derivative.

		Returns
		-------
		Array containing the partial x derivative vector components.
		"""
		return np.array([0, 0])

	def partial_z(self, x, z, t):
		r"""
		Computes the partial derivative of the fluid **u** = (_u_, _w_) with
		respect to the vertical position _z_.

		Parameters
		----------
		x : float or array
			The x position(s) at which to evaluate the derivative.
		z : float or array
			The z position(s) at which to evaluate the derivative.
		t : float or array
			The time(s) at which to evaluate the derivative.

		Returns
		-------
		Array containing the partial z derivative vector components.
		"""
		return np.array([0, 0])

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
		return np.array([0, 0])

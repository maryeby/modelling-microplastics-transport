import numpy as np
import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
from transport_framework import flow

class RotatingFlow(flow.Flow):
	"""Represents a fluid flow for rigid body rotation."""

	def __init__(self, depth=15):
		r"""
		Attributes
		----------
		depth : float, default=15
			The depth of the fluid *h*.
		period : float
			A parameter used in the computation of the timespan over which to
			integrate.
		"""
		self.depth = depth
		self.period = 1

	def velocity(self, x, z, t=None):
		"""
		Computes the fluid velocity _**u**_.

		Parameters
		----------
		x : float or array
			The x position(s) at which to evaluate the velocity.
		z : float or array
			The z position(s) at which to evaluate the velocity.
		t : float or array, default=None
			The time(s) at which to evaluate the velocity.

		Returns
		-------
		Array containing the velocity field vector components *u* and *w*.
		"""
		return np.array([-z, x])

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
		return np.array([0, 1])

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
		return np.array([-1, 0])

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

	def velocity(self, x, z):
		"""
		Computes the fluid velocity _**u**_.

		Parameters
		----------
		x : float or array
			The x position(s) at which to evaluate the velocity.
		z : float or array
			The z position(s) at which to evaluate the velocity.

		Returns
		-------
		Array containing the velocity field vector components *u* and *w*.
		"""
		return np.array([-z, x])

	def material_derivative(self, x, z): 
		"""
		Computes the Lagrangian derivative.

		Parameters
		----------
		x : float or array
			The x position(s) at which to evaluate the fluid velocity.
		z : float or array
			The z position(s) at which to evaluate the velocity and derivative.

		Returns
		-------
		Array containing the material derivative vector components.
		"""
		return np.array([-x, -z])

	def material_derivative2(self, x, z):
		"""
		Computes the second order Lagrangian derivative.

		Parameters
		----------
		x : float or array
			The x position(s) at which to evaluate the fluid velocity.
		z : float or array
			The z position(s) at which to evaluate the velocity and derivative.

		Returns
		-------
		Array containing the second order material derivative vector components.
		"""
		return np.array([z, -x])

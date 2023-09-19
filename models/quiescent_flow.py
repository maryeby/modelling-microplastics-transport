import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
from transport_framework import flow

class QuiescentFlow(flow.Flow):
	"""Represents a quiescent fluid flow."""

	def __init__(self, depth=15):
		r"""
		Attributes
		----------
		depth : float, default=15
			The depth of the fluid *h*.
		gravity : float
			The gravity **g** acting on the fluid.
		period : float
			A parameter used in the computation of the integration timespan.
		"""
		super().__init__(depth)
	
	def set_gravity(self):
		r"""
		Defines the gravity vector **g** as,
		$$\mathbf{g} = \langle 0, -1 \rangle.$$
		"""
		self.gravity = np.array([0, -1])
	
	def velocity(self, x, z, t):
		r"""
		Computes the fluid velocity as, $$\mathbf{u} = (u, w) = \mathbf{0}.$$

		Parameters
		----------
		x : float or array
			The horizontal position(s) at which to evaluate the velocity.
		z : float or array
			The vertical position(s) at which to evaluate the velocity.
		t : float or array
			The time(s) at which to evaluate the velocity.

		Returns
		-------
		Array containing the velocity field vector components *u* and *w*.
		"""
		return np.array([0, 0])

	def partial_t(self, x, z, t):
		r"""
		Computes the partial derivative of the fluid with respect to time as,
		$$\frac{\partial \mathbf{u}}{\partial t} = \mathbf{0}.$$

		Parameters
		----------
		x : float or array
			The horizontal position(s) at which to evaluate the derivative.
		z : float or array
			The vertical position(s) at which to evaluate the derivative.
		t : float or array
			The time(s) at which to evaluate the derivative.

		Returns
		-------
		Array containing the vector components of the derivative.
		"""
		return np.array([0, 0])

	def partial_x(self, x, z, t):
		r"""
		Computes the partial derivative of the fluid with respect to the
		horizontal position as,
		$$\frac{\partial \mathbf{u}}{\partial x} = \mathbf{0}.$$

		Parameters
		----------
		x : float or array
			The horizontal position(s) at which to evaluate the derivative.
		z : float or array
			The vertical position(s) at which to evaluate the derivative.
		t : float or array
			The time(s) at which to evaluate the derivative.

		Returns
		-------
		Array containing the vector components of the derivative.
		"""
		return np.array([0, 0])

	def partial_z(self, x, z, t):
		r"""
		Computes the partial derivative of the fluid with respect to the
		vertical position as,
		$$\frac{\partial \mathbf{u}}{\partial z} = \mathbf{0}.$$

		Parameters
		----------
		x : float or array
			The horizontal position(s) at which to evaluate the derivative.
		z : float or array
			The vertical position(s) at which to evaluate the derivative.
		t : float or array
			The time(s) at which to evaluate the derivative.

		Returns
		-------
		Array containing the vector components of the derivative.
		"""
		return np.array([0, 0])

	def material_derivative2(self, x, z, t):
		r"""
		Computes the second order material derivative as,
		$$\frac{\mathrm{D}^2 \mathbf{u}}{\mathrm{D} t^2} = \mathbf{0}.$$

		Parameters
		----------
		x : float or array
			The horizontal position(s) at which to evaluate the derivative.
		z : float or array
			The vertical position(s) at which to evaluate the derivative.
		t : float or array
			The time(s) at which to evaluate the derivative.

		Returns
		-------
		Array containing the second order material derivative vector components.
		"""
		return np.array([0, 0])

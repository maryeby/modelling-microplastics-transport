import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
from transport_framework import flow

class RotatingFlow(flow.Flow):
	"""Represents a fluid flow for rigid body rotation."""

	def __init__(self, depth=50):
		r"""
		Attributes
		----------
		depth : float, default=15
			The depth of the fluid *h*.
		gravity : array
			The gravity **g** acting on the fluid.
		period : float
			A parameter used in the computation of the timespan over which to
			integrate.
		"""
		super().__init__(depth)
		self.gravity = np.array([0, 0]) # gravity not considered for this flow

	def velocity(self, x, z, t=None):
		r"""
		Computes the fluid velocity as,
		$$\textbf{u} = (u, w) = \langle -z, x \rangle.$$

		Parameters
		----------
		x : float or array
			The horizontal position(s) at which to evaluate the velocity.
		z : float or array
			The vertical position(s) at which to evaluate the velocity.
		t : float or array, default=None
			The time(s) at which to evaluate the velocity.

		Returns
		-------
		Array containing the velocity field vector components *u* and *w*.
		"""
		return np.array([-z, x])

	def partial_t(self, x, z, t):
		r"""
		Computes the partial derivative of the fluid with respect to time as,
		$$\frac{\partial \textbf{u}}{\partial t} = \mathbf{0}.$$

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
		return np.array([np.zeros(x.shape), np.zeros(z.shape)])

	def partial_x(self, x, z, t): 
		r"""
		Computes the partial derivative of the fluid with respect to the
		horizontal position as,
		$$\frac{\partial \textbf{u}}{\partial x} = \langle 0, 1 \rangle.$$

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
		return np.array([np.zeros(x.shape), np.ones(z.shape)])

	def partial_z(self, x, z, t): 
		r"""
		Computes the partial derivative of the fluid with respect to the
		vertical positiona as,
		$$\frac{\partial \textbf{u}}{\partial z} = \langle -1, 0 \rangle.$$

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
		return np.array([-np.ones(x.shape), np.zeros(z.shape)])

import numpy as np
from transport_framework import flow

class RotatingFlow(flow.Flow):
	"""Represent a fluid flow for rigid body rotation.[^1]"""

	def __init__(self, depth=50):
		r"""
		Attributes
		----------
		depth : float, default=50
			The depth of the fluid *h*.
		gravity : ndarray
			1D array of `float` data, the gravity ***g*** acting on the fluid.
		period : float
			A parameter used in the computation of the timespan over which to
			integrate.
			
		References
		----------
		[^1]: [A. Daitche (2013).](https://doi.org/10.1016/j.jcp.2013.07.024)
			  Advection of inertial particles in the presence of the history
			  force: Higher order numerical schemes. *Journal of Computational
			  Physics* 254, 93–106.
		"""
		super().__init__(depth)
		self.gravity = np.array([0, 0]) # gravity not considered for this flow

	def velocity(self, x, z, t=None):
		r"""
		Compute the fluid velocity,
		$$\boldsymbol{u} = \langle u, w \rangle = \langle -z, x \rangle.$$

		Parameters
		----------
		x, z : float or ndarray
			The horizontal and vertical position(s).
		t : float or ndarray, default=None
			The time(s) at which to evaluate the velocity.

		Returns
		-------
		ndarray
			1D array of `float` data, the vector components *u* and *w*.
		"""
		return np.array([-z, x])

	def partial_t(self, x, z, t):
		r"""
		Computes the partial derivative of the fluid with respect to time as,
		$$\frac{\partial \boldsymbol{u}}{\partial t} = \mathbf{0}.$$

		Parameters
		----------
		x, z : float or ndarray
			The horizontal and vertical position(s).
		t : float or ndarray
			The time(s) at which to evaluate the derivative.

		Returns
		-------
		ndarray
			1D array of `float` data, the vector components of the derivative.
		"""
		return np.array([np.zeros(x.shape), np.zeros(z.shape)])

	def partial_x(self, x, z, t): 
		r"""
		Computes the partial derivative of the fluid with respect to the
		horizontal position as,
		$$\frac{\partial \boldsymbol{u}}{\partial x} = \langle 0, 1 \rangle.$$

		Parameters
		----------
		x, z : float or ndarray
			The horizontal and vertical position(s).
		t : float or ndarray
			The time(s) at which to evaluate the derivative.

		Returns
		-------
		ndarray
			1D array of `float` data, the vector components of the derivative.
		"""
		return np.array([np.zeros(x.shape), np.ones(z.shape)])

	def partial_z(self, x, z, t): 
		r"""
		Computes the partial derivative of the fluid with respect to the
		vertical position as,
		$$\frac{\partial \boldsymbol{u}}{\partial z} = \langle -1, 0 \rangle.$$

		Parameters
		----------
		x, z : float or ndarray
			The horizontal and vertical position(s).
		t : float or ndarray
			The time(s) at which to evaluate the derivative.

		Returns
		-------
		ndarray
			1D array of `float` data, the vector components of the derivative.
		"""
		return np.array([-np.ones(x.shape), np.zeros(z.shape)])

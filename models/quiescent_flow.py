import numpy as np
from transport_framework import flow

class QuiescentFlow(flow.Flow):
	"""Represent a quiescent fluid flow."""

	def __init__(self, depth=50):
		r"""
		Attributes
		----------
		depth : float, default=50
			The depth of the fluid *h*.
		gravity : ndarray
			1D array of `float` data, the gravity ***g*** acting on the fluid.
		period : float
			A parameter used in the computation of the integration timespan.
		"""
		super().__init__(depth)
		self.gravity = np.array([0, -1])
	
	def velocity(self, x, z, t):
		r"""
		Compute the fluid velocity,
		$$\boldsymbol{u} = \langle u, w \rangle = \mathbf{0}.$$

		Parameters
		----------
		x, z : float or ndarray
			The horizontal and vertical position(s).
		t : float or ndarray
			The time(s) at which to evaluate the velocity.

		Returns
		-------
		ndarray
			1D array of `float` data, the vector components *u* and *w*.
		"""
		if isinstance(t, np.ndarray):
			return np.zeros((2, t.size))
		else:
			return np.zeros((2,))

	def partial_t(self, x, z, t):
		r"""
		Compute the partial derivative of the fluid with respect to time,
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
		if isinstance(t, np.ndarray):
			return np.zeros((2, t.size))
		else:
			return np.zeros((2,))

	def partial_x(self, x, z, t):
		r"""
		Compute the partial derivative of the fluid with respect to the
		horizontal position,
		$$\frac{\partial \boldsymbol{u}}{\partial x} = \mathbf{0}.$$

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
		if isinstance(t, np.ndarray):
			return np.zeros((2, t.size))
		else:
			return np.zeros((2,))

	def partial_z(self, x, z, t):
		r"""
		Compute the partial derivative of the fluid with respect to the vertical
		position, $$\frac{\partial \boldsymbol{u}}{\partial z} = \mathbf{0}.$$

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
		if isinstance(t, np.ndarray):
			return np.zeros((2, t.size))
		else:
			return np.zeros((2,))

	def material_derivative2(self, x, z, t):
		r"""
		Compute the second order material derivative,
		$$\frac{\mathrm{D}^2 \boldsymbol{u}}{\mathrm{D} t^2} = \mathbf{0}.$$

		Parameters
		----------
		x, z : float or ndarray
			The horizontal and vertical position(s).
		t : float or ndarray
			The time(s) at which to evaluate the derivative.

		Returns
		-------
		ndarray
			1D array of `float` data, the vector components of the solution.
		"""
		if isinstance(t, np.ndarray):
			return np.zeros((2, t.size))
		else:
			return np.zeros((2,))

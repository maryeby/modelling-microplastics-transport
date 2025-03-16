from abc import ABC, abstractmethod
import numpy as np
import scipy.constants as constants

class Flow(ABC):
	"""Represent a fluid flow."""

	def __init__(self, depth):
		r"""
		Attributes
		----------
		depth : float
			The depth of the fluid *h'*.
		gravity : ndarray
			1D array of `float` data, the gravity **g'** acting on the fluid.
		period : float
			A parameter used in the computation of the integration timespan.
		"""
		self.depth = depth
		self.period = 1
		self.gravity = np.array([0, -constants.g])

	@abstractmethod
	def velocity(self, x, z, t):
		r"""
		Compute the fluid velocity, $$\mathbf{u} = \langle u, w \rangle.$$

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
		pass

	@abstractmethod
	def partial_t(self, x, z, t): 
		r"""
		Compute the partial derivative of the fluid with respect to time,
		$$\frac{\partial \mathbf{u}}{\partial t}.$$

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
		pass

	@abstractmethod
	def partial_x(self, x, z, t): 
		r"""
		Compute the partial derivative of the fluid with respect to the
		horizontal position, $$\frac{\partial \mathbf{u}}{\partial x}.$$

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
		pass

	@abstractmethod
	def partial_z(self, x, z, t):
		r"""
		Compute the partial derivative of the fluid with respect to the
		vertical position, $$\frac{\partial \mathbf{u}}{\partial z}.$$

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
		pass

	def dot_jacobian(self, vec, x, z, t):
		r"""
		Compute the dot product of `vec` with the Jacobian of the fluid,
		$$\texttt{vec} \cdot \nabla \mathbf{u}.$$

		Parameters
		----------
		vec: ndarray
			1D array of `float` data, the vector to be used in the dot product.
		x, z : float or ndarray
			The horizontal and vertical position(s).
		t : float or ndarray
			The time(s) at which to evaluate the solution.

		Returns
		-------
		ndarray
			1D array of `float` data, the vector components of the solution.
		"""
		x_component, z_component = vec
		dxu, dxw = self.partial_x(x, z, t)
		dzu, dzw = self.partial_z(x, z, t)
		return np.array([dxu * x_component + dzu * z_component,
						 dxw * x_component + dzw * z_component])

	def material_derivative(self, x, z, t): 
		r"""
		Compute the material derivative,
		$$\frac{\mathrm{D}\mathbf{u}}{\mathrm{D}t}
			= \frac{\partial \mathbf{u}}{\partial t}
			+ \mathbf{u} \cdot \nabla \mathbf{u}.$$

		Parameters
		----------
		x, z : float or ndarray
			The horizontal and vertical position(s).
		t : float or ndarray
			The time(s) at which to evaluate the derivative.

		Returns
		-------
		ndarray
			1D array of `float` data, the material derivative vector components.
		"""
		return self.partial_t(x, z, t) \
			   + self.dot_jacobian(self.velocity(x, z, t), x, z, t)

	def derivative_along_trajectory(self, x, z, t, v):
		r"""
		Compute the derivative of the fluid along the particle trajectory,
		$$\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}t}
			= \frac{\partial \mathbf{u}}{\partial t}
			+ \mathbf{v} \cdot \nabla \mathbf{u}.$$

		Parameters
		----------
		x, z : float or ndarray
			The horizontal and vertical position(s) of the particle.
		t : float or ndarray
			The time(s) at which to evaluate the derivative.
		v : ndarray
			1D array of `float` data, the particle velocity.

		Returns
		-------
		ndarray
			1D array of `float` data, the vector components of the solution.
		"""
		return self.partial_t(x, z, t) + self.dot_jacobian(v, x, z, t)

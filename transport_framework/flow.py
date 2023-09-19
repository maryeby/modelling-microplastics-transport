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
		gravity : array
			The gravity **g** acting on the fluid.
		period : float
			A parameter used in the computation of the integration timespan.
		"""
		self.depth = depth
		self.period = 1
		self.set_gravity()

	@abstractmethod
	def set_gravity(self):
		"""Defines the gravity vector **g**."""
		pass

	@abstractmethod
	def velocity(self, x, z, t):
		r"""
		Computes the fluid velocity, $$\mathbf{u} = (u, w).$$

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
		pass

	@abstractmethod
	def partial_t(self, x, z, t): 
		r"""
		Computes the partial derivative of the fluid with respect to time,
		$$\frac{\partial \mathbf{u}}{\partial t}.$$

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
		pass

	@abstractmethod
	def partial_x(self, x, z, t): 
		r"""
		Computes the partial derivative of the fluid with respect to the
		horizontal position, $$\frac{\partial \mathbf{u}}{\partial x}.$$

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
		pass

	@abstractmethod
	def partial_z(self, x, z, t):
		r"""
		Computes the partial derivative of the fluid with respect to the
		vertical position, $$\frac{\partial \mathbf{u}}{\partial z}.$$

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
		pass

	def dot_jacobian(self, vec, x, z, t):
		r"""
		Computes the dot product of the provided vector with the Jacobian of the
		fluid, $$\texttt{vec} \cdot \nabla \mathbf{u}.$$

		Parameters
		----------
		vec: array
			The vector to be used in the dot product.
		x : float or array
			The horizontal position(s) at which to evaluate the solution.
		z : float or array
			The vertical position(s) at which to evaluate the solution.
		t : float or array
			The time(s) at which to evaluate the solution.

		Returns
		-------
		Array containing the vector components of the dot product.
		"""
		x_component, z_component = vec
		dxu, dxw = self.partial_x(x, z, t)
		dzu, dzw = self.partial_z(x, z, t)
		return np.array([dxu * x_component + dzu * z_component,
						 dxw * x_component + dzw * z_component])

	def material_derivative(self, x, z, t): 
		r"""
		Computes the material derivative,
		$$\frac{\mathrm{D}\mathbf{u}}{\mathrm{D}t}
			= \frac{\partial \mathbf{u}}{\partial t}
			+ \mathbf{u} \cdot \nabla \mathbf{u}.$$

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
		Array containing the material derivative vector components.
		"""
		return self.partial_t(x, z, t) \
			   + self.dot_jacobian(self.velocity(x, z, t), x, z, t)

	def derivative_along_trajectory(self, x, z, t, v):
		r"""
		Computes the derivative of the fluid along the provided trajectory of
		the particle,
		$$\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}t}
			= \frac{\partial \mathbf{u}}{\partial t}
			+ \mathbf{v} \cdot \nabla \mathbf{u}.$$

		Parameters
		----------
		x : float or array
			The horizontal position(s) at which to evaluate the derivative.
		z : float or array
			The vertical position(s) at which to evaluate the derivative.
		t : float or array
			The time(s) at which to evaluate the derivative.
		v : array
			The velocity of the particle.

		Returns
		-------
		Array containing the derivative vector components.
		"""
		return self.partial_t(x, z, t) + self.dot_jacobian(v, x, z, t)

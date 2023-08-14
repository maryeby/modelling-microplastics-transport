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
		period : float
			A parameter used in the computation of the timespan over which to
			integrate.
		"""
		self.depth = depth
		self.period = 1

	@abstractmethod
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
		pass

	@abstractmethod
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
		pass

	@abstractmethod
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
		pass

	@abstractmethod
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
		pass

	def dot_jacobian(self, vec, x, z, t):
		r"""
		Computes the dot product of the provided vector with the Jacobian of the
		fluid **u**.

		Parameters
		----------
		vec: array
			The vector to be used in the dot product.
		x : float or array
			The x position(s) at which to evaluate the solution.
		z : float or array
			The z position(s) at which to evaluate the solution.
		t : float or array
			The time(s) at which to evaluate the solution.

		Returns
		-------
		Array containing the vector components of the resulting dot product.
		"""
		x_component, z_component = vec
		dxu, dxw = self.partial_x(x, z, t)
		dzu, dzw = self.partial_z(x, z, t)
		return np.array([dxu * x_component + dzu * z_component,
						 dxw * x_component + dzw * z_component])

	def material_derivative(self, x, z, t): 
		r"""
		Computes the material derivative,
		$$\frac{\mathrm{D}\textbf{u}}{\mathrm{D}t}
			= \frac{\partial \textbf{u}}{\partial t}
			+ \textbf{u} \cdot \nabla \textbf{u}.$$

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
		return self.partial_t(x, z, t) \
			   + self.dot_jacobian(self.velocity(x, z, t), x, z, t)

	def derivative_along_trajectory(self, x, z, t, v):
		r"""
		Computes the derivative of the fluid along the provided trajectory of
		the particle,
		$$\frac{\mathrm{d}\textbf{u}}{\mathrm{d}t}
			= \frac{\partial \textbf{u}}{\partial t}
			+ \textbf{v} \cdot \nabla \textbf{u}.$$

		Parameters
		----------
		x : float or array
			The x position(s) at which to evaluate the fluid velocity.
		z : float or array
			The z position(s) at which to evaluate the fluid velocity.
		t : float or array
			The time(s) at which to evaluate the fluid velocity.
		v : array
			The velocity of the particle.

		Returns
		-------
		Array
			The horizontal and vertical components of the solution.
		"""
		return self.velocity(x, z, t) + self.dot_jacobian(v, x, z, t)

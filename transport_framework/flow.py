from abc import ABC, abstractmethod
import numpy as np

class Flow(ABC):
	"""Represents a fluid flow."""

	def __init__(self, amplitude, wavelength, depth, density=1):
		r"""
		Attributes
		----------
		amplitude : float
			The amplitude of the wave *A*.
		wavelength : float
			The wavelength lambda.
		depth : float
			The depth of the fluid *h*.
		wavenum : float
			The wave number *k*, computed as $$k = \frac{2 \pi}{\lambda}.$$
		gravity : float
			The gravity _**g**_.
		angular_freq : float
			The angular frequency omega, computed using the dispersion relation.
		phase_velocity : float
			The phase velocity *c*.
		period : float
			The period of the wave, computed as
			$$\text{period} = \frac{2\pi}{\omega}.$$
		"""
		self.amplitude = amplitude
		self.wavelength = wavelength
		self.depth = depth

		# computed attributes
		self.wavenum = 2 * np.pi / self.wavelength
		self.set_angular_freq()
		self.set_gravity()
		self.phase_velocity = self.angular_freq / self.wavenum
		self.period = 2 * np.pi / self.angular_freq
		
	@abstractmethod
	def set_angular_freq(self):
		"""Defines the angular frequency omega."""
		pass

	@abstractmethod
	def set_gravity(self):
		"""Defines the gravity _**g**_."""
		pass

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
	def material_derivative(self, x, z, t):
		"""
		Computes the Lagrangian derivative.

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
		pass

	@abstractmethod
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
		pass

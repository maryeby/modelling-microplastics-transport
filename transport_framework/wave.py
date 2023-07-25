import numpy as np
import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
from abc import ABC, abstractmethod
from transport_framework import flow

class Wave(flow.Flow):
	"""Represents a fluid flow."""

	def __init__(self, depth, amplitude, wavelength):
		r"""
		Attributes
		----------
		depth : float
			The depth of the fluid *h*.
		amplitude : float
			The amplitude of the wave *A*.
		wavelength : float
			The wavelength lambda.
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
		super().__init__(depth)
		self.amplitude = amplitude
		self.wavelength = wavelength

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

import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
from abc import ABC, abstractmethod
from transport_framework import flow

class Wave(flow.Flow):
	"""Represents a fluid wave."""

	def __init__(self, depth, amplitude, wavelength):
		r"""
		Attributes
		----------
		depth : float
			The depth of the fluid *h*.
		amplitude : float
			The amplitude of the wave *A*.
		wavelength : float
			The wavelength *λ*.
		wavenum : float
			The wavenumber *k*, computed as $$k = \frac{2 \pi}{\lambda}.$$
		gravity : float
			The gravity **g** acting on the fluid.
		angular_freq : float
			The angular frequency *ω*, computed using the dispersion relation.
		phase_velocity : float
			The phase velocity *c*, computed as $$c = \frac{\omega}{k}.$$
		period : float
			The period of the wave, computed as
			$$\text{period} = \frac{2\pi}{\omega}.$$
		max_velocity : float
			The maximum velocity *U* at the surface z = 0.
		froude_num : float
			The Froude number *Fr*, computed as $$Fr = \frac{U}{c}.$$
		"""
		super().__init__(depth)
		self.amplitude = amplitude
		self.wavelength = wavelength

		# computed attributes
		self.wavenum = 2 * np.pi / self.wavelength
		self.set_angular_freq()
		self.phase_velocity = self.angular_freq / self.wavenum
		self.period = 2 * np.pi / self.angular_freq
		self.max_velocity = self.angular_freq * self.amplitude
		self.froude_num = self.max_velocity / self.phase_velocity
		
	@abstractmethod
	def set_angular_freq(self):
		"""Defines the angular frequency *ω*."""
		pass

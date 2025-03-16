import numpy as np
import scipy.constants as constants
from abc import ABC, abstractmethod
from transport_framework import flow

class Wave(flow.Flow):
	"""Represent a wavy flow."""

	def __init__(self, depth, amplitude, wavelength):
		r"""
		Attributes
		----------
		depth : float
			The depth of the fluid *h'*.
		amplitude : float
			The amplitude of the wave *A'*.
		wavelength : float
			The wavelength *λ'*.
		kinematic_viscosity : float
			The kinematic viscosity ν' of seawater.
		wavenum : float
			The wavenumber *k'*, computed as $$k' = \frac{2 \pi}{\lambda'}.$$
		gravity : ndarray
			1D array of `float` data, the gravity **g'** acting on the fluid.
		angular_freq : float
			The angular frequency *ω'*, computed using the dispersion relation.
		phase_velocity : float
			The phase velocity *c'*, computed as $$c' = \frac{\omega'}{k'}.$$
		period : float
			The period of the wave, computed as
			$$\text{period}' = \frac{2\pi}{\omega'}.$$
		froude_num : float
			The Froude number *Fr*, computed as
			$$Fr = \sqrt{\frac{k'(\omega'A')^2}{g'}}.$$
		reynolds_num : float
			The Reynolds number *Re* of the wave, computed as
			$$Re = \frac{\omega'A'}{k'ν'}.$$
		"""
		super().__init__(depth)
		self.amplitude = amplitude
		self.wavelength = wavelength
		self.kinematic_viscosity = 1e-6

		# computed attributes
		self.wavenum = 2 * np.pi / self.wavelength
		self.set_angular_freq()
		self.phase_velocity = self.angular_freq / self.wavenum
		self.period = 2 * np.pi / self.angular_freq
		self.froude_num = np.sqrt(self.wavenum 
						* (self.angular_freq * self.amplitude) ** 2
						/ constants.g)
		self.reynolds_num = self.angular_freq * self.amplitude / (self.wavenum
											  * self.kinematic_viscosity)
	
	@abstractmethod
	def set_angular_freq(self):
		"""Define the angular frequency *ω'*."""
		pass

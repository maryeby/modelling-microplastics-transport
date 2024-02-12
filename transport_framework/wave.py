import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
import scipy.constants as constants
from abc import ABC, abstractmethod
from transport_framework import flow

class Wave(flow.Flow):
	"""Represents a fluid wave."""

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
		gravity : float
			The gravity **g'** acting on the fluid.
		angular_freq : float
			The angular frequency *ω'*, computed using the dispersion relation.
		phase_velocity : float
			The phase velocity *c'*, computed as $$c' = \frac{\omega'}{k'}.$$
		period : float
			The period of the wave, computed as
			$$\text{period}' = \frac{2\pi}{\omega'}.$$
		max_velocity : float
			The maximum velocity *U'* at the surface *z'* = 0, computed as
			$$U' = \omega' A'.$$
		froude_num : float
			The Froude number *Fr*, computed as
			$$Fr = \sqrt{\frac{k'U'^2}{g'}}.$$
		reynolds_num : float
			The Reynolds number *Re* of the wave, computed as
			$$Re = \frac{U'}{k'ν'}.$$
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
		self.max_velocity = self.angular_freq * self.amplitude
		self.froude_num = np.sqrt(self.wavenum * self.max_velocity ** 2
											   / constants.g)
		self.reynolds_num = self.max_velocity / (self.wavenum
											  * self.kinematic_viscosity)
	
	@abstractmethod
	def set_angular_freq(self):
		"""Defines the angular frequency *ω'*."""
		pass

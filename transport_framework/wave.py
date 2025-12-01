import numpy as np
import scipy.constants as constants
from abc import ABC, abstractmethod
from transport_framework import flow

class Wave(flow.Flow):
	"""Represent an oscillatory flow with gravitational effects."""

	def __init__(self, depth, amplitude, wavelength):
		r"""
		Attributes
		----------
		depth : float
			The depth of the fluid *h'*.
		amplitude : float
			The amplitude of the wave *A'*.
		wavelength : float
			The wavelength $\lambda'$.
		kinematic_viscosity : float
			The kinematic viscosity $\nu'$ of seawater.
		wavenum : float
			The wavenumber, $$k' = 2 \pi / \lambda'.$$
		steepness : float
			The wave steepness, $$\epsilon = k'A'.$$
		gravity : ndarray
			1D array of `float` data, the gravity ***g'*** acting on the fluid.
		angular_freq : float
			The angular frequency $\omega'$, computed using the dispersion
			relation.
		phase_velocity : float
			The phase velocity, $$c' = \omega' / k'.$$
		period : float
			The wave period, where $$\text{period}' = 2\pi / \omega'.$$
		froude_num : float
			The Froude number, $$Fr = \omega' / \sqrt{g'k'}.$$
		reynolds_num : float
			The Reynolds number of the wave, $$Re = \frac{\omega'}{k^{\prime 2}
			\nu}'.$$
		"""
		super().__init__(depth)
		self.amplitude = amplitude
		self.wavelength = wavelength
		self.kinematic_viscosity = 1e-6

		# computed attributes
		self.wavenum = 2 * np.pi / wavelength
		self.steepness = self.wavenum * amplitude
		self.set_angular_freq()
		self.phase_velocity = self.angular_freq / self.wavenum
		self.period = 2 * np.pi / self.angular_freq
		self.froude_num = self.angular_freq / np.sqrt(constants.g
											* self.wavenum)
		self.reynolds_num = self.angular_freq / (self.wavenum * self.wavenum
											  * self.kinematic_viscosity)
	
	@abstractmethod
	def set_angular_freq(self):
		r"""Define the angular frequency $\omega'$."""
		pass

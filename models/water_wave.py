import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
from scipy import constants
from transport_framework import wave

class WaterWave(wave.Wave):
	"""
	Represents a non-dimensional linear water wave with arbitrary depth.
	"""

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
			The gravity **g** acting on the fluid, non-dimensionalized as,
			$$g' = \frac{g}{k'U'^{\prime 2}}.$$
		angular_freq : float
			The angular frequency *ω'*, computed using the dispersion relation,
			$$\omega' = \sqrt{g'k' \tanh(k'h')}.$$
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
		super().__init__(depth, amplitude, wavelength)
		self.gravity /= self.wavenum * self.max_velocity ** 2

	def set_angular_freq(self):
		r"""
		Defines the angular frequency omega with the dispersion relation,
		$$\omega' = \sqrt{g'k' \tanh(k'h')}.$$
		"""
		k, h = self.wavenum, self.depth
		self.angular_freq = np.sqrt(constants.g * k * np.tanh(k * h))

	def velocity(self, x, z, t):
		r"""
		Computes the fluid velocity, $$\mathbf{u} = \langle u, w \rangle,$$
		$$u(x, z, t) = \frac{\cosh(z + h)}{\cosh(h)}
					   \cos\Bigg(x - \frac{t}{k'A'}\Bigg),$$
		$$w(x, z, t) = \frac{\sinh(z + h)}{\cosh(h)}
					   \sin\Bigg(x - \frac{t}{k'A'}\Bigg).$$

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
		k, A = self.wavenum, self.amplitude
		h = k * self.depth
		return np.array([np.cosh(z + h) / np.cosh(h) * np.cos(x - t / (k * A)),
						 np.sinh(z + h) / np.cosh(h) * np.sin(x - t / (k * A))])

	def partial_t(self, x, z, t): 
		r"""
		Computes the partial derivative of the fluid with respect to time,
		$$\frac{\partial \mathbf{u}}{\partial t} =
			\Bigg\langle \frac{1}{k'A'} \frac{\cosh(z + h)}{\cosh(h)}
						 \sin\Bigg(x - \frac{t}{k'A'}\Bigg), \;
						-\frac{1}{k'A'} \frac{\sinh(z + h)}{\cosh(h)}
						 \cos\Bigg(x - \frac{t}{k'A'}\Bigg)\Bigg\rangle.$$

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
		k, A = self.wavenum, self.amplitude
		h = k * self.depth
		return np.array([np.cosh(z + h) / np.cosh(h) \
										* np.sin(x - t / (k * A)) / (k * A),
						-np.sinh(z + h) / np.cosh(h) \
										* np.cos(x - t / (k * A)) / (k * A)])

	def partial_x(self, x, z, t): 
		r"""
		Computes the partial derivative of the fluid with respect to the
		horizontal position,
		$$\frac{\partial \mathbf{u}}{\partial x} =
			\Bigg\langle -\frac{\cosh(z + h)}{\cosh(h)}
						 \sin\Bigg(x - \frac{t}{k'A'}\Bigg), \;
						 \frac{\sinh(z + h)}{\cosh(h)}
						 \cos\Bigg(x - \frac{t}{k'A'}\Bigg)\Bigg\rangle.$$

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
		k, A = self.wavenum, self.amplitude
		h = k * self.depth
		return np.array([-np.cosh(z + h) / np.cosh(h) * np.sin(x - t / (k * A)),
						 np.sinh(z + h) / np.cosh(h) * np.cos(x - t / (k * A))])

	def partial_z(self, x, z, t):
		r"""
		Computes the partial derivative of the fluid with respect to the
		vertical position,
		$$\frac{\partial \mathbf{u}}{\partial z} =
			\Bigg\langle \frac{\sinh(z + h)}{\cosh(h)}
						 \sin\Bigg(x - \frac{t}{k'A'}\Bigg), \;
						 \frac{\cosh(z + h)}{\cosh(h)}
						 \cos\Bigg(x - \frac{t}{k'A'}\Bigg)\Bigg\rangle.$$

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
		k, A = self.wavenum, self.amplitude
		h = k * self.depth
		return np.array([np.sinh(z + h) / np.cosh(h) * np.cos(x - t / (k * A)),
						 np.cosh(z + h) / np.cosh(h) * np.sin(x - t / (k * A))])

import numpy as np
from scipy import constants
from transport_framework import wave
from utils.colors import print_warning

class WaterWave(wave.Wave):
	"""Represent a non-dimensional linear water wave with arbitrary depth."""

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
			$$g' = \frac{g}{k'(\omega'A')^2}.$$
		angular_freq : float
			The angular frequency *ω'*, computed using the dispersion relation,
			$$\omega' = \sqrt{g'k' \tanh(k'h')}.$$
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
		super().__init__(depth, amplitude, wavelength)
		self.gravity /= self.wavenum * (self.angular_freq * self.amplitude) ** 2
		self.period *= self.angular_freq * self.wavenum * self.amplitude
		if 0.1 * np.tanh(self.wavenum * self.depth) < self.wavenum \
													* self.amplitude:
			print_warning('Wave steepness parameter '
				+ f'(epsilon = {self.wavenum * self.amplitude:.4f}) is not '
				+ f'<< tanh(h) (= {np.tanh(self.wavenum * self.depth):.4f}).')

	def set_angular_freq(self):
		r"""
		Define the angular frequency omega with the dispersion relation,
		$$\omega' = \sqrt{g'k' \tanh(k'h')}.$$
		"""
		k, h = self.wavenum, self.depth
		self.angular_freq = np.sqrt(constants.g * k * np.tanh(k * h))

	def velocity(self, x, z, t):
		r"""
		Compute the fluid velocity, $$\mathbf{u} = \langle u, w \rangle,$$
		$$u(x, z, t) = \frac{\cosh(z + h)}{\sinh(h)}
					   \cos\Bigg(x - \frac{t}{\epsilon}\Bigg),$$
		$$w(x, z, t) = \frac{\sinh(z + h)}{\sinh(h)}
					   \sin\Bigg(x - \frac{t}{\epsilon}\Bigg),$$
		where, $$\epsilon = k'A'.$$

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
		k = self.wavenum
		h = k * self.depth
		epsilon = k * self.amplitude # wave steepness
		return np.array([np.cosh(z + h) / np.sinh(h) * np.cos(x - t / epsilon),
						 np.sinh(z + h) / np.sinh(h) * np.sin(x - t / epsilon)])

	def partial_t(self, x, z, t): 
		r"""
		Compute the partial derivative of the fluid with respect to time,
		$$\frac{\partial \mathbf{u}}{\partial t} =
			\Bigg\langle \frac{1}{\epsilon} \frac{\cosh(z + h)}{\sinh(h)}
						 \sin\Bigg(x - \frac{t}{\epsilon}\Bigg), \;
						-\frac{1}{\epsilon} \frac{\sinh(z + h)}{\sinh(h)}
						 \cos\Bigg(x - \frac{t}{\epsilon}\Bigg)\Bigg\rangle.$$

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
		k = self.wavenum
		h = k * self.depth
		epsilon = k * self.amplitude # wave steepness
		return np.array([np.cosh(z + h) / np.sinh(h) * np.sin(x - t / epsilon) \
										/ epsilon,
						-np.sinh(z + h) / np.sinh(h) * np.cos(x - t / epsilon) \
										/ epsilon])

	def partial_x(self, x, z, t): 
		r"""
		Compute the partial derivative of the fluid with respect to the
		horizontal position,
		$$\frac{\partial \mathbf{u}}{\partial x} =
			\Bigg\langle -\frac{\cosh(z + h)}{\sinh(h)}
						 \sin\Bigg(x - \frac{t}{\epsilon}\Bigg), \;
						 \frac{\sinh(z + h)}{\sinh(h)}
						 \cos\Bigg(x - \frac{t}{\epsilon}\Bigg)\Bigg\rangle.$$

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
		k = self.wavenum
		h = k * self.depth
		epsilon = k * self.amplitude # wave steepness
		return np.array([-np.cosh(z + h) / np.sinh(h) * np.sin(x - t / epsilon),
						 np.sinh(z + h) / np.sinh(h) * np.cos(x - t / epsilon)])

	def partial_z(self, x, z, t):
		r"""
		Compute the partial derivative of the fluid with respect to the
		vertical position,
		$$\frac{\partial \mathbf{u}}{\partial z} =
			\Bigg\langle \frac{\sinh(z + h)}{\sinh(h)}
						 \cos\Bigg(x - \frac{t}{\epsilon}\Bigg), \;
						 \frac{\cosh(z + h)}{\sinh(h)}
						 \sin\Bigg(x - \frac{t}{\epsilon}\Bigg)\Bigg\rangle.$$

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
		k = self.wavenum
		h = k * self.depth
		epsilon = k * self.amplitude # wave steepness
		return np.array([np.sinh(z + h) / np.sinh(h) * np.cos(x - t / epsilon),
						 np.cosh(z + h) / np.sinh(h) * np.sin(x - t / epsilon)])

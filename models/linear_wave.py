import numpy as np
from scipy import constants
from transport_framework import wave
from utils.colors import print_warning

class LinearWave(wave.Wave):
	"""Represent a dimensionless linear water wave with arbitrary depth."""

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
		gravity : float
			The gravity **g** acting on the fluid, non-dimensionalized as,
			$$\boldsymbol{g} = \frac{\boldsymbol{g}'k'}{\omega^{\prime 2}}.$$
		angular_freq : float
			The angular frequency $\omega'$, computed using the dispersion
			relation, $$\omega' = \sqrt{g'k' \tanh(k'h')}.$$
		phase_velocity : float
			The phase velocity, $$c' = \omega' / k'.$$
		period : float
			The period of the wave,
			$$\text{period}' = \frac{2\pi}{\omega'},$$
			and non-dimensionalized as
			$$\text{period} = \text{period}'\omega'.$$
		froude_num : float
			The Froude number $$Fr = \frac{\omega'}{\sqrt{g'k'}}.$$
		reynolds_num : float
			The Reynolds number of the wave, $$Re = \frac{\omega'}{k^{\prime 2}
			\nu'}.$$
		"""
		super().__init__(depth, amplitude, wavelength)
		self.gravity *= self.wavenum / (self.angular_freq * self.angular_freq)
		self.period *= self.angular_freq
		if 0.1 * np.tanh(self.wavenum * self.depth) < np.round(self.wavenum \
			   * self.amplitude, 5):
			print_warning('Wave steepness parameter '
				+ f'(epsilon = {self.wavenum * self.amplitude:.5f}) is not '
				+ f'<< tanh(h) (= {np.tanh(self.wavenum * self.depth):g}).')

	def set_angular_freq(self):
		r"""
		Define the angular frequency omega with the dispersion relation,
		$$\omega' = \sqrt{g'k' \tanh(k'h')}.$$
		"""
		k, h = self.wavenum, self.depth
		self.angular_freq = np.sqrt(constants.g * k * np.tanh(k * h))

	def velocity(self, x, z, t):
		r"""
		Compute the fluid velocity, $$\boldsymbol{u} = \langle u, w \rangle,$$
		$$u(x, z, t) = \epsilon \frac{\cosh(z + h)}{\sinh(h)} \cos(x - t),$$
		$$w(x, z, t) = \epsilon \frac{\sinh(z + h)}{\sinh(h)} \sin(x - t).$$

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
		epsilon = self.steepness
		return np.array([epsilon * np.cosh(z + h) / np.sinh(h) * np.cos(x - t),
						 epsilon * np.sinh(z + h) / np.sinh(h) * np.sin(x - t)])

	def partial_t(self, x, z, t): 
		r"""
		Compute the partial derivative of the fluid with respect to time,
		$$\frac{\partial \boldsymbol{u}}{\partial t} =
			\Bigg\langle \epsilon \frac{\cosh(z + h)}{\sinh(h)} \sin(x - t), \;
						-\epsilon \frac{\sinh(z + h)}{\sinh(h)}
						 \cos(x - t)\Bigg\rangle.$$

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
		epsilon = self.steepness
		return np.array([epsilon * np.cosh(z + h) / np.sinh(h) * np.sin(x - t),
						-epsilon * np.sinh(z + h) / np.sinh(h) * np.cos(x - t)])

	def partial_x(self, x, z, t): 
		r"""
		Compute the partial derivative of the fluid with respect to the
		horizontal position,
		$$\frac{\partial \boldsymbol{u}}{\partial x} = \Bigg\langle -\epsilon
						 \frac{\cosh(z + h)}{\sinh(h)} \sin(x - t), \;
						 \epsilon \frac{\sinh(z + h)}{\sinh(h)}
						 \cos(x - t)\Bigg\rangle.$$

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
		epsilon = self.steepness
		return np.array([-epsilon * np.cosh(z + h) / np.sinh(h) * np.sin(x - t),
						 epsilon * np.sinh(z + h) / np.sinh(h) * np.cos(x - t)])

	def partial_z(self, x, z, t):
		r"""
		Compute the partial derivative of the fluid with respect to the
		vertical position,
		$$\frac{\partial \boldsymbol{u}}{\partial z} = \Bigg\langle \epsilon
						 \frac{\sinh(z + h)}{\sinh(h)} \cos(x - t), \;
						 \epsilon \frac{\cosh(z + h)}{\sinh(h)}
						 \sin(x - t)\Bigg\rangle.$$

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
		epsilon = self.steepness # wave steepness
		return np.array([epsilon * np.sinh(z + h) / np.sinh(h) * np.cos(x - t),
						 epsilon * np.cosh(z + h) / np.sinh(h) * np.sin(x - t)])

import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
from scipy import constants
from transport_framework import wave

class DimensionalWaterWave(wave.Wave):
	"""
	Represents a dimensional linear water wave with arbitrary depth.
	"""

	def __init__(self, amplitude, wavelength, depth):
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
			The angular frequency *ω*, computed using the dispersion relation,
			$$\omega = \sqrt{g \tanh(kh)}.$$
		phase_velocity : float
			The phase velocity *c*, computed as $$c = \frac{\omega}{k}.$$
		period : float
			The period of the wave, computed as
			$$\text{period} = \frac{2\pi}{\omega}.$$
		max_velocity : float
			The maximum velocity *U* at the surface z = 0, computed as
			$$U = \omega A.$$
		froude_num : float
			The Froude number *Fr*, computed as $$Fr = \frac{U}{c}.$$
		"""
		super().__init__(depth, amplitude, wavelength)

	def set_gravity(self):
		r"""
		Defines the gravity vector **g** as,
		$$\mathbf{g} = \langle 0, -g \rangle.$$
		"""
		self.gravity = np.array([0, -constants.g])

	def set_angular_freq(self):
		r"""
		Defines the angular frequency omega with the dispersion relation,
		$$\omega = \sqrt{g \tanh(kh)}.$$
		"""
		self.angular_freq = np.sqrt(constants.g
							* np.tanh(self.wavenum * self.depth))

	def velocity(self, x, z, t):
		r"""
		Computes the fluid velocity, $$\mathbf{u} = \langle u, w \rangle,$$
		$$u(x, z, t) = U\frac{\cosh(k(z + h))}{\sinh(kh)}\sin(\omega t - kx),$$
		$$w(x, z, t) = U\frac{\sinh(k(z + h))}{\sinh(kh)}\cos(\omega t - kx).$$

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
		U = self.max_velocity
		k = self.wavenum
		h = self.depth
		omega = self.angular_freq

		return np.array([U * np.cosh(k * (z + h)) / np.sinh(k * h)
						   * np.sin(omega * t - k * x),
						 U * np.sinh(k * (z + h)) / np.sinh(k * h)
						   * np.cos(omega * t - k * x)])

	def partial_t(self, x, z, t): 
		r"""
		Computes the partial derivative of the fluid with respect to time,
		$$\frac{\partial \mathbf{u}}{\partial t} =
			\Bigg\langle \omega U \frac{\cosh(k(z + h))}{\sinh(kh)}
						 \cos(\omega t - kx), \;
						-\omega U\frac{\sinh(k(z + h))}{\sinh(kh)}
						 \sin(\omega t - kx)\Bigg\rangle.$$

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
		U = self.max_velocity
		k = self.wavenum
		h = self.depth
		omega = self.angular_freq
		return np.array([omega * U * np.cosh(k * (z + h)) / np.sinh(k * h)
							   * np.cos(omega * t - k * x),
						 -omega * U * np.sinh(k * (z + h)) / np.sinh(k * h)
								* np.sin(omega * t - k * x)])

	def partial_x(self, x, z, t): 
		r"""
		Computes the partial derivative of the fluid with respect to the
		horizontal position,
		$$\frac{\partial \mathbf{u}}{\partial x} =
			\Bigg\langle -kU \frac{\cosh(k(z + h))}{\sinh(kh)}
						 \cos(\omega t - kx), \;
						 kU \frac{\sinh(k(z + h))}{\sinh(kh)}
						 \sin(\omega t - kx)\Bigg\rangle.$$

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
		U = self.max_velocity
		k = self.wavenum
		h = self.depth
		omega = self.angular_freq
		return np.array([-k * U * np.cosh(k * (z + h)) / np.sinh(k * h)
							* np.cos(omega * t - k * x),
						 k * U * np.sinh(k * (z + h)) / np.sinh(k * h)
						   * np.sin(omega * t - k * x)])

	def partial_z(self, x, z, t):
		r"""
		Computes the partial derivative of the fluid with respect to the
		vertical position,
		$$\frac{\partial \mathbf{u}}{\partial z} =
			\Bigg\langle kU \frac{\sinh(k(z + h))}{\sinh(kh)}
						 \sin(\omega t - kx), \;
						 kU \frac{\cosh(k(z + h))}{\sinh(kh)}
						 \cos(\omega t - kx)\Bigg\rangle.$$

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
		U = self.max_velocity
		k = self.wavenum
		h = self.depth
		omega = self.angular_freq
		return np.array([k * U * np.sinh(k * (z + h)) / np.sinh(k * h)
						   * np.sin(omega * t - k * x),
						 k * U * np.cosh(k * (z + h)) / np.sinh(k * h)
						   * np.cos(omega * t - k * x)])

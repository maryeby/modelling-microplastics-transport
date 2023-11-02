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
			The gravity **g** acting on the fluid, non-dimensionalized as,
			$$g' = g \frac{L'}{U^{\prime 2}}.$$
		angular_freq : float
			The angular frequency *ω*, computed using the dispersion relation,
			$$\omega = \sqrt{gk \tanh(kh)}.$$
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
		self.gravity /= constants.g * self.froude_num ** 2

	def set_angular_freq(self):
		r"""
		Defines the angular frequency omega with the dispersion relation,
		$$\omega = \sqrt{gk \tanh(kh)}.$$
		"""
		k, h = self.wavenum, self.depth
		self.angular_freq = np.sqrt(constants.g * k * np.tanh(k * h))

	def velocity(self, x, z, t):
		r"""
		Computes the fluid velocity, $$\mathbf{u} = \langle u, w \rangle,$$
		$$u(x, z, t) = \frac{\cosh(z + h)}{\cosh(h)}
					   \cos\Bigg(x - \frac{t}{Fr}\Bigg),$$
		$$w(x, z, t) = \frac{\sinh(z + h)}{\cosh(h)}
					   \sin\Bigg(x - \frac{t}{Fr}\Bigg).$$

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
		h, Fr = self.depth, self.froude_num
		return np.array([np.cosh(z + h) / np.cosh(h) * np.cos(x - t / Fr),
						 np.sinh(z + h) / np.cosh(h) * np.sin(x - t / Fr)])

	def partial_t(self, x, z, t): 
		r"""
		Computes the partial derivative of the fluid with respect to time,
		$$\frac{\partial \mathbf{u}}{\partial t} =
			\Bigg\langle \frac{1}{Fr} \frac{\cosh(z + h)}{\cosh(h)}
						 \sin\Bigg(x - \frac{t}{Fr}\Bigg), \;
						-\frac{1}{Fr} \frac{\sinh(z + h)}{\cosh(h)}
						 \cos\Bigg(x - \frac{t}{Fr}\Bigg)\Bigg\rangle.$$

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
		h, Fr = self.depth, self.froude_num
		return np.array([np.cosh(z + h) / np.cosh(h) * np.sin(x - t / Fr) / Fr,
						-np.sinh(z + h) / np.cosh(h) * np.cos(x - t / Fr) / Fr])

	def partial_x(self, x, z, t): 
		r"""
		Computes the partial derivative of the fluid with respect to the
		horizontal position,
		$$\frac{\partial \mathbf{u}}{\partial x} =
			\Bigg\langle -\frac{\cosh(z + h)}{\cosh(h)}
						 \sin\Bigg(x - \frac{t}{Fr}\Bigg), \;
						 \frac{\sinh(z + h)}{\cosh(h)}
						 \cos\Bigg(x - \frac{t}{Fr}\Bigg)\Bigg\rangle.$$

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
		h, Fr = self.depth, self.froude_num
		return np.array([-np.cosh(z + h) / np.cosh(h) * np.sin(x - t / Fr),
						  np.sinh(z + h) / np.cosh(h) * np.cos(x - t / Fr)])

	def partial_z(self, x, z, t):
		r"""
		Computes the partial derivative of the fluid with respect to the
		vertical position,
		$$\frac{\partial \mathbf{u}}{\partial z} =
			\Bigg\langle \frac{\sinh(z + h)}{\cosh(h)}
						 \sin\Bigg(x - \frac{t}{Fr}\Bigg), \;
						 \frac{\cosh(z + h)}{\cosh(h)}
						 \cos\Bigg(x - \frac{t}{Fr}\Bigg)\Bigg\rangle.$$

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
		h, Fr = self.depth, self.froude_num
		return np.array([np.sinh(z + h) / np.cosh(h) * np.cos(x - t / Fr),
						 np.cosh(z + h) / np.cosh(h) * np.sin(x - t / Fr)])

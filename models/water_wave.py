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
		self.gravity /= self.wavenum * self.max_velocity ** 2

	def set_gravity(self):
		r"""
		Defines the gravity **g** as,
		$$\mathbf{g} = \langle 0, g \rangle,$$
		which is non-dimensionalized as,
		$$\mathbf{g} = \Bigg\langle 0, \frac{g}{kU^2} \Bigg\rangle.$$
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
		$$u(x, z, t) = \frac{\cosh(z + h)}{\sinh(h)}
					   \sin\Bigg(\frac{t}{Fr} - x\Bigg),$$
		$$w(x, z, t) = \frac{\sinh(z + h)}{\sinh(h)}
					   \cos\Bigg(\frac{t}{Fr} - x\Bigg).$$

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
		h = self.depth
		Fr = self.froude_num
		return np.array([np.cosh(z + h) / np.sinh(h) * np.sin(t / Fr - x),
						 np.sinh(z + h) / np.sinh(h) * np.cos(t / Fr - x)])

	def partial_t(self, x, z, t): 
		r"""
		Computes the partial derivative of the fluid with respect to time,
		$$\frac{\partial \mathbf{u}}{\partial t} =
			\Bigg\langle \frac{1}{Fr} \frac{\cosh(z + h)}{\sinh(h)}
						 \cos\Bigg(\frac{t}{Fr} - x\Bigg), \;
						-\frac{1}{Fr} \frac{\sinh(z + h)}{\sinh(h)}
						 \sin\Bigg(\frac{t}{Fr} - x\Bigg)\Bigg\rangle.$$

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
		h = self.depth
		Fr = self.froude_num
		return np.array([np.cosh(z + h) / np.sinh(h) * np.cos(t / Fr - x) / Fr,
						-np.sinh(z + h) / np.sinh(h) * np.sin(t / Fr - x) / Fr])

	def partial_x(self, x, z, t): 
		r"""
		Computes the partial derivative of the fluid with respect to the
		horizontal position,
		$$\frac{\partial \mathbf{u}}{\partial x} =
			\Bigg\langle -\frac{\cosh(z + h)}{\sinh(h)}
						 \cos\Bigg(\frac{t}{Fr} - x\Bigg), \;
						 \frac{\sinh(z + h)}{\sinh(h)}
						 \sin\Bigg(\frac{t}{Fr} - x\Bigg)\Bigg\rangle.$$

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
		h = self.depth
		Fr = self.froude_num
		return np.array([-np.cosh(z + h) / np.sinh(h) * np.cos(t / Fr - x),
						  np.sinh(z + h) / np.sinh(h) * np.sin(t / Fr - x)])

	def partial_z(self, x, z, t):
		r"""
		Computes the partial derivative of the fluid with respect to the
		vertical position,
		$$\frac{\partial \mathbf{u}}{\partial z} =
			\Bigg\langle \frac{\sinh(z + h)}{\sinh(h)}
						 \sin\Bigg(\frac{t}{Fr} - x\Bigg), \;
						 \frac{\cosh(z + h)}{\sinh(h)}
						 \cos\Bigg(\frac{t}{Fr} - x\Bigg)\Bigg\rangle.$$

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
		h = self.depth
		Fr = self.froude_num
		return np.array([np.sinh(z + h) / np.sinh(h) * np.sin(t / Fr - x),
						 np.cosh(z + h) / np.sinh(h) * np.cos(t / Fr - x)])

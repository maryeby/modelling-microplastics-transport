import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
from scipy import constants
from transport_framework import wave

class DeepWaterWave(wave.Wave):
	"""
	Represents a non-dimensional version of the fluid flow described in
	Santamaria et al., (2013). The flow is a linear water wave with infinite 
	depth.
	"""

	def __init__(self, amplitude, wavelength, depth=50):
		r"""
		Attributes
		----------
		depth : float, default=50
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
			$$\omega = \sqrt{gk}.$$
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
		$$\omega = \sqrt{gk}.$$
		"""
		self.angular_freq = np.sqrt(constants.g * self.wavenum)

	def velocity(self, x, z, t):
		r"""
		Computes the fluid velocity, $$\textbf{u} = \langle u, w \rangle,$$
		$$u(x, z, t) = e^{z} \cos\Bigg(x - \frac{t}{Fr}\Bigg),$$
		$$w(x, z, t) = e^{z} \sin\Bigg(x - \frac{t}{Fr}\Bigg).$$

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
		return np.array([np.exp(z) * np.cos(x - t / self.froude_num),
						 np.exp(z) * np.sin(x - t / self.froude_num)])

	def partial_t(self, x, z, t):
		r"""
		Computes the partial derivative of the fluid with respect to time,
		$$\frac{\partial \mathbf{u}}{\partial t} =
		\Bigg\langle \frac{w}{Fr}, \; -\frac{u}{Fr} \Bigg\rangle.$$

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
		u, w = self.velocity(x, z, t)
		return np.array([w / self.froude_num, -u / self.froude_num])

	def partial_x(self, x, z, t):
		r"""
		Computes the partial derivative of the fluid with respect to the
		horizontal position,
		$$\frac{\partial \mathbf{u}}{\partial x} = \langle -w, \; u \rangle.$$

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
		u, w = self.velocity(x, z, t)
		return np.array([-w, u])

	def partial_z(self, x, z, t):
		r"""
		Computes the partial derivative of the fluid with respect to the
		vertical position,
		$$\frac{\partial \mathbf{u}}{\partial z} = \langle u, \; w \rangle.$$

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
		return self.velocity(x, z, t)

	def material_derivative2(self, x, z, t):
		r"""
		Computes the second order material derivative, where
		$$\frac{\mathrm{D}^2\textbf{u}}{\mathrm{D}t^2} =
		\Bigg\langle e^{2z} - \frac{u}{Fr^2}, \quad
		w \Bigg(2 e^{2z} - \frac{1}{Fr^2}\Bigg) \Bigg\rangle.$$

		Parameters
		----------
		x : float or array
			The x position(s) at which to evaluate the fluid velocity.
		z : float or array
			The z position(s) at which to evaluate the velocity and derivative.
		t : float or array
			The time(s) at which to evaluate the velocity.

		Returns
		-------
		Array containing the second order material derivative vector components.
		"""
		u, w = self.velocity(x, z, t)
		Fr = self.froude_num
		return np.array([np.exp(2 * z) / Fr - u / Fr ** 2,
						 w * (2 * np.exp(2 * z) - 1 / Fr ** 2)])

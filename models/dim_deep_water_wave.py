import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
from scipy import constants
from transport_framework import wave

class DimensionalDeepWaterWave(wave.Wave):
	"""
	Represents the fluid flow described in Santamaria et al., (2013). The flow
	is a linear water wave with infinite depth.
	"""

	def __init__(self, amplitude, wavelength, depth=15):
		r"""
		Attributes
		----------
		depth : float, default=15
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

	def set_gravity(self):
		r"""
		Defines the gravity vector **g** as,
		$$\mathbf{g} = \langle 0, -g \rangle.$$
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
		$$u(x, z, t) = Ue^{kz} \cos(kx - \omega t),$$
		$$w(x, z, t) = Ue^{kz} \sin(kx - \omega t).$$

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
		omega = self.angular_freq

		return np.array([U * np.exp(k * z) * np.cos(k * x - omega * t), 
						 U * np.exp(k * z) * np.sin(k * x - omega * t)])

	def partial_t(self, x, z, t): 
		r"""
		Computes the partial derivative of the fluid with respect to time,
		$$\frac{\partial \mathbf{u}}{\partial t} =
		\langle \omega w, \; -\omega u \rangle.$$

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
		omega = self.angular_freq
		u, w = self.velocity(x, z, t)
		return np.array([omega * w, omega * -u])

	def partial_x(self, x, z, t): 
		r"""
		Computes the partial derivative of the fluid with respect to the
		horizontal position,
		$$\frac{\partial \mathbf{u}}{\partial x} = \langle -kw, \; ku \rangle.$$

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
		k = self.wavenum
		u, w = self.velocity(x, z, t)
		return np.array([k * -w, k * u])

	def partial_z(self, x, z, t):
		r"""
		Computes the partial derivative of the fluid with respect to the
		vertical position,
		$$\frac{\partial \mathbf{u}}{\partial z} = \langle ku, \; kw \rangle.$$

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
		return self.wavenum * self.velocity(x, z, t)

	def material_derivative2(self, x, z, t):
		r"""
		Computes the second order material derivative, where
		$$\frac{\mathrm{D}^2\textbf{u}}{\mathrm{D}t^2} =
		\langle U^2 e^{2kz} \omega k - \omega^2 u, \quad
		w(2 e^{2kz} U^2 k^2 - \omega^2) \rangle.$$

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
		U = self.max_velocity
		k = self.wavenum
		epsilon = self.amplitude * k
		omega = self.angular_freq
		u, w = self.velocity(x, z, t)

		return np.array([k * w * U ** 2 * np.exp(2 * k * z) - omega ** 2 * u,
						 w * (2 * np.exp(2 * k * z) * (U * k) ** 2
						   - omega ** 2)])

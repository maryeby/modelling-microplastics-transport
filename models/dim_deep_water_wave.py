import numpy as np
import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
from scipy import constants
from transport_framework import flow

class DimensionalDeepWaterWave(flow.Flow):
	"""
	Represents the fluid flow described in Santamaria et al., (2013). The flow
	is a linear water wave with infinite depth.
	"""

	def __init__(self, amplitude, wavelength, depth=15, density=1):
		r"""
		Attributes
		----------
		amplitude : float
			The amplitude of the wave *A*.
		wavelength : float
			The wavelength lambda.
		depth : float, default=15
			The depth of the fluid *h*.
		density : float
			The density of the fluid rho_f.
		wavenum : float
			The wave number *k*, computed as $$k = \frac{2 \pi}{\lambda}.$$
		gravity : float
			The gravity _**g**_.
		angular_freq : float
			The angular frequency omega, computed using the dispersion relation.
		max_velocity : float
			The maximum velocity *U* at the surface z = 0.
		period : float
			The period of the wave, computed as
			$$\text{period} = \frac{2\pi}{\omega}.$$
		"""
		super().__init__(amplitude, wavelength, depth, density)
		self.max_velocity = self.angular_freq * self.amplitude

	def set_gravity(self):
		"""Defines the gravity _**g**_."""
		self.gravity = np.array([0, -constants.g])

	def set_angular_freq(self):
		r"""
		Defines the angular frequency omega with the dispersion relation,
		$$\omega = \sqrt{gk}.$$
		"""
		self.angular_freq = np.sqrt(constants.g * self.wavenum)

	def velocity(self, x, z, t):
		r"""
		Computes the fluid velocity _**u**_ = (_u_, _w_), with
		$$u(x, z, t) = Ue^{kz} \cos(kx - \omega t),$$
		$$w(x, z, t) = Ue^{kz} \sin(kx - \omega t).$$

		Parameters
		----------
		x : float or array
			The x position(s) at which to evaluate the velocity.
		z : float or array
			The z position(s) at which to evaluate the velocity.
		t : float or array
			The time(s) at which to evaluate the velocity.

		Returns
		-------
		Array containing the velocity field vector components _u_ and _w_.
		"""
		U = self.max_velocity
		k = self.wavenum
		omega = self.angular_freq

		return np.array([U * np.exp(k * z) * np.cos(k * x - omega * t), 
						 U * np.exp(k * z) * np.sin(k * x - omega * t)])

	def material_derivative(self, x, z, t):
		r"""
		Computes the Lagrangian derivative, where
		$$\Bigg[\frac{\mathrm{D}\textbf{u}}{\mathrm{D}t}\Bigg]_x = \omega w,
			\qquad
		\Bigg[\frac{\mathrm{D}\textbf{u}}{\mathrm{D}t}\Bigg]_z
			= U^2 k e^{2kz} - \omega u.$$

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
		Array containing the material derivative vector components.
		"""
		U = self.max_velocity
		k = self.wavenum
		epsilon = self.amplitude * k
		omega = self.angular_freq
		u, w = self.velocity(x, z, t)
		return np.array([omega * w, k * U ** 2 * np.exp(2 * k * z) - omega * u])

	def material_derivative2(self, x, z, t):
		r"""
		Computes the second order Lagrangian derivative, where
		$$\Bigg[\frac{\mathrm{D}^2\textbf{u}}{\mathrm{D}t^2}\Bigg]_x
			= U^2 e^{2kz} \omega k - \omega^2 u,
			\qquad
		\Bigg[\frac{\mathrm{D}^2\textbf{u}}{\mathrm{D}t^2}\Bigg]_z
			= w(2 e^{2kz} U^2 k^2 - \omega^2).$$

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

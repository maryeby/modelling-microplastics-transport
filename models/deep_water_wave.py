import numpy as np
import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
from scipy import constants
from transport_framework import flow

class DeepWaterWave(flow.Flow):
	"""
	Represents a non-dimensional version of the fluid flow described in
	Santamaria et al., (2013). The flow is a linear water wave with infinite 
	depth.
	"""

	def __init__(self, amplitude, wavelength, depth=15, density=1):
		r"""
		Attributes
		----------
		amplitude : float
			The amplitude of the wave *A*.
		wavelength : float
			The wavelength lambda.
		depth : float=15
			The depth of the fluid *h*.
		density : float
			The density of the fluid rho_f.
		wavenum : float
			The wave number *k*, computed as $$k = \frac{2 \pi}{\lambda}.$$
		gravity : float
			The gravity _**g**_.
		angular_freq : float
			The angular frequency omega, computed using the dispersion relation.
		period : float
			The period of the wave, computed as
			$$\text{period} = \frac{2\pi}{\omega}.$$
		epsilon : float
			A relationship between the wave ampltiude *A* and wave number *k*.
		"""
		super().__init__(amplitude, wavelength, depth, density)
		self.epsilon = self.amplitude * self.wavenum

	def set_angular_freq(self):
		r"""
		Defines the angular frequency omega with the dispersion relation,
		$$\omega = \sqrt{gk}.$$
		"""
		self.angular_freq = np.sqrt(constants.g * self.wavenum)

	def set_gravity(self):
		"""Defines the gravity _**g**_."""
		self.gravity = np.array([0, -constants.g]) / (self.wavenum \
					   * (self.angular_freq * self.amplitude) ** 2)

	def velocity(self, x, z, t):
		r"""
		Computes the fluid velocity _**u**_ = (_u_, _w_), with
		$$u(x, z, t) = e^{z} \cos(x - \frac{t}{\epsilon}),$$
		$$w(x, z, t) = e^{z} \sin(x - \frac{t}{\epsilon}).$$

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
		return np.array([np.exp(z) * np.cos(x - t / self.epsilon),
                         np.exp(z) * np.sin(x - t / self.epsilon)])

	def material_derivative(self, x, z, t):
		r"""
		Computes the Lagrangian derivative, where
		$$\Bigg[\frac{\mathrm{D}\textbf{u}}{\mathrm{D}t}\Bigg]_x =
			\frac{w}{\epsilon}, \qquad
		\Bigg[\frac{\mathrm{D}\textbf{u}}{\mathrm{D}t}\Bigg]_z
			= e^{2z} - \frac{u}{\epsilon}.$$

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
		u, w = self.velocity(x, z, t)
		return np.array([w / self.epsilon, np.exp(2 * z) - u / self.epsilon])

	def material_derivative2(self, x, z, t):
		r"""
		Computes the second order Lagrangian derivative, where
		$$\Bigg[\frac{\mathrm{D}^2\textbf{u}}{\mathrm{D}t^2}\Bigg]_x
			= \frac{e^{2z}}{\epsilon} - \frac{u}{\epsilon^2},
			\qquad
		\Bigg[\frac{\mathrm{D}^2\textbf{u}}{\mathrm{D}t^2}\Bigg]_z
			= w(2e^{2z} - \frac{1}{\epsilon^2}).$$

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
		return np.array([np.exp(2 * z) - u / self.epsilon ** 2,
                         w * (2 * np.exp(2 * z) - 1 / self.epsilon ** 2)])

import numpy as np
from scipy import constants
from transport_framework import wave
from utils.colors import print_warning

class BichromaticWave(wave.Wave):
	"""Represent a dimensionless bichromatic wave of arbitrarily deep water."""

	def __init__(self, depth, amplitude, wavelength):
		r"""
		Attributes
		----------
		depth : float
			The depth of the fluid *h'*.
		amplitude : ndarray
			1D array of `float` values, the wave amplitudes $A'_1$ and $A'_2$.
		wavelength : ndarray
			1D array of `float` values, the wavelengths $\lambda'_1$ and
			$\lambda'_2$.
		kinematic_viscosity : float
			The kinematic viscosity ν' of seawater.
		wavenum : ndarray
			1D array of `float` values, the wavenumbers $k'_1$ and $k'_2$,
			computed as $$k'_i = \frac{2 \pi}{\lambda'_i}.$$
		gravity : float
			The gravity **g** acting on the fluid, non-dimensionalized as,
			$$g' = \frac{g}{k'_1(\omega'_1 A'_1)^2}.$$
		angular_freq : ndarray
			1D array of `float` values, the angular frequencies $\omega'_1$ and
			$\omega'_2$, computed using the dispersion relation,
			$$\omega'_i = \sqrt{g'k'_i \tanh(k'_i h')}.$$
		phase_velocity : ndarray
			 1D array of `float` values, the phase velocities $c'_1$ and $c'_2$,
			 computed as $$c'_i = \frac{\omega'_i}{k'_i}.$$
		period : ndarray
			1D array of `float` values, the periods of the wave, computed as
			$$\text{period}'_i = \frac{2\pi}{\omega'_i},$$
			and non-dimensionalized as
			$$\text{period} = \text{period}' * k'_1 \omega'_1 A'_1.$$
		froude_num : float
			1D array of `float` values, the Froude numbers $Fr_1$ and $Fr_2$,
			computed as $$Fr_i = \sqrt{\frac{k'_i (\omega'_i A'_i)^2}{g'}}.$$
		reynolds_num : float
			1D array of `float` values, the Reynolds numbers $Re_1$ and $Re_2$
			of the wave, computed as $$Re_i = \frac{\omega'_i A'_i}{k'_i ν'}.$$
		"""
		super().__init__(depth, amplitude, wavelength)
		self.gravity /= self.wavenum[0] * (self.angular_freq[0]
										* self.amplitude[0]) ** 2
		self.period *= self.angular_freq[0] * self.wavenum[0] \
											* self.amplitude[0]
	def set_angular_freq(self):
		r"""
		Define the angular frequency omega with the dispersion relation,
		$$\omega'_i = \sqrt{g'k'_i \tanh(k'_i h')}.$$
		"""
		k, h = self.wavenum, self.depth
		self.angular_freq = np.sqrt(constants.g * k * np.tanh(k * h))

	def velocity(self, x, z, t):
		r"""
		Compute the fluid velocity $\mathbf{u} = \langle u, w \rangle,$
		$$u(x, z, t) = \frac{\cosh(z + h)}{\sinh(h)} \cos(x - t / \epsilon)
					 + \psi \frac{\cosh(\kappa (z + h))}{\sinh(\kappa h)}
					   \cos(\kappa x - \tau t / \epsilon)\\
					 + K_u \zeta \epsilon g \cos(\xi x - \chi t / \epsilon
					 + \phi_u) \frac{cc_g - 2 (c_g^2 + gh)}{2ch(gh - c_g^2)},$$
		$$w(x, z, t) = \frac{\sinh(z + h)}{\sinh(h)} \sin(x - t / \epsilon)
					 + \psi \frac{\sinh(\kappa (z + h))}{\sinh(\kappa h)}
					   \sin(\kappa x - \tau t / \epsilon),$$
		where, $$\epsilon = k'_1 A'_1,$$ $$\kappa = \frac{k'_2}{k'_1},$$
			   $$\tau = \frac{\omega'_2}{\omega'_1},$$
			   $$\psi = \frac{\omega'_2 A'_2}{\omega'_1 A'_1},$$
			   $$\xi = \frac{k'_g}{k'_1},$$
			   $$\chi = \frac{\omega'_g}{\omega'_1},$$
			   $$\zeta = k'_1 A'_2,$$ and $K_u = 0$ for a flat seabed.

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
		# dimensionless parameters
		h = self.wavenum[0] * self.depth
		epsilon = self.wavenum[0] * self.amplitude[0]
		kappa = self.wavenum[1] / self.wavenum[0]
		tau = self.angular_freq[1] / self.angular_freq[0]
		psi = self.angular_freq[1] * self.amplitude[1] / (self.angular_freq[0]
								   * self.amplitude[0])

		# velocity field components
		u = np.cosh(z + h) * np.cos(x - t / epsilon) / np.sinh(h) \
					  + psi * np.cosh(kappa * (z + h)) \
					  * np.cos(kappa * x - tau * t / epsilon) \
					  / np.sinh(kappa * h)
		w = np.sinh(z + h) * np.sin(x - t / epsilon) / np.sinh(h) \
					  + psi * np.sinh(kappa * (z + h)) * np.sin(kappa * x - tau
					  * t / epsilon) / np.sinh(kappa * h)
		return np.array([u, w])

	def partial_t(self, x, z, t): 
		r"""
		Compute the partial derivative of the fluid with respect to time,
		$$\frac{\partial u}{\partial t} = \frac{1}{\epsilon}
			\Bigg(\frac{\cosh(z + h)}{\sinh(h)} \sin(x - t / \epsilon)
			+ \psi \tau \frac{\cosh(\kappa (z + h))}{\sinh{\kappa h}}
			  \sin(\kappa x - \tau t / \epsilon)\Bigg)\\+ K_u \zeta \chi g
			  \sin(\xi x - \chi t / \epsilon + \phi)
			  \frac{cc_g - 2 (c_g^2 + gh)}{2ch (gh - c_g^2)},$$
		$$\frac{\partial w}{\partial t} = -\frac{1}{\epsilon}
			\Bigg(\frac{\sinh(z + h)}{\sinh(h)} \cos(x - t / \epsilon)
			+ \psi \tau \frac{\sinh(\kappa (z + h))}{\sinh(\kappa h)}
			  \cos(\kappa x - \tau t / \epsilon)\Bigg).$$

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
		# dimensionless parameters
		h = self.wavenum[0] * self.depth
		epsilon = self.wavenum[0] * self.amplitude[0]
		kappa = self.wavenum[1] / self.wavenum[0]
		tau = self.angular_freq[1] / self.angular_freq[0]
		psi = self.angular_freq[1] * self.amplitude[1] / (self.angular_freq[0]
								   * self.amplitude[0])

		# partial derivative components
		dudt = (np.cosh(z + h) * np.sin(x - t / epsilon) / np.sinh(h) \
			 + psi * tau * np.cosh(kappa * (z + h)) * np.sin(kappa * x - tau * t
			 / epsilon) / np.sinh(kappa * h)) / epsilon
		dwdt = -(np.sinh(z + h) * np.cos(x - t / epsilon) / np.sinh(h) \
			 + psi * tau * np.sinh(kappa * (z + h)) * np.cos(kappa * x - tau * t
			 / epsilon) / np.sinh(kappa * h)) / epsilon
		return np.array([dudt, dwdt])

	def partial_x(self, x, z, t): 
		r"""
		Compute the partial derivative of the fluid with respect to the
		horizontal position,
		$$\frac{\partial u}{\partial x} = -\frac{\cosh(z + h)}{\sinh(h)}
			\sin(x - t / \epsilon) - \psi \kappa \frac{\cosh(\kappa (z + h))}
			{\sinh(\kappa h)} \sin(\kappa x - \tau t / \epsilon)\\
			- K_u \zeta \xi \epsilon g \sin(\xi x - \chi t / \epsilon + \phi)
			\frac{cc_g - 2 (c_g^2 + gh)}{2ch(gh - c_g^2)},$$
		$$\frac{\partial w}{\partial x} = \frac{\sinh(z + h)}{\sinh(h)}
			\cos(x - t / \epsilon) + \psi \kappa \frac{\sinh(\kappa (z + h))}
			{\sinh(\kappa h)} \cos(\kappa x - \tau t / \epsilon).$$

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
		# dimensionless parameters
		h = self.wavenum[0] * self.depth
		epsilon = self.wavenum[0] * self.amplitude[0]
		kappa = self.wavenum[1] / self.wavenum[0]
		tau = self.angular_freq[1] / self.angular_freq[0]
		psi = self.angular_freq[1] * self.amplitude[1] / (self.angular_freq[0]
								   * self.amplitude[0])

		# partial derivative components
		dudx = -np.cosh(z + h) * np.sin(x - t / epsilon) / np.sinh(h) - psi \
			 * kappa * np.cosh(kappa * (z + h)) * np.sin(kappa * x - tau * t 
			 / epsilon) / np.sinh(kappa * h)
		dwdx = np.sinh(z + h) * np.cos(x - t / epsilon) / np.sinh(h) + psi \
			 * kappa * np.sinh(kappa * (z + h)) * np.cos(kappa * x - tau * t \
			 / epsilon) / np.sinh(kappa * h)
		return np.array([dudx, dwdx])

	def partial_z(self, x, z, t):
		r"""
		Compute the partial derivative of the fluid with respect to the
		vertical position,
		$$\frac{\partial u}{\partial z} = \frac{\sinh(z + h)}{\sinh(h)}
			\cos(x - t / \epsilon) + \psi \kappa \frac{\sinh(\kappa (z + h))}
			{\sinh(\kappa h)} \cos(\kappa x - \tau t / \epsilon),$$
		$$\frac{\partial w}{\partial z} = \frac{\cosh(z + h)}{\sinh(h)}
			\sin(x - t / \epsilon) + \psi \kappa \frac{\cosh(\kappa (z + h))}
			{\sinh(\kappa h)} \sin(\kappa x - \tau t / \epsilon).$$

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
		# dimensionless parameters
		h = self.wavenum[0] * self.depth
		epsilon = self.wavenum[0] * self.amplitude[0]
		kappa = self.wavenum[1] / self.wavenum[0]
		tau = self.angular_freq[1] / self.angular_freq[0]
		psi = self.angular_freq[1] * self.amplitude[1] / (self.angular_freq[0]
								   * self.amplitude[0])

		# partial derivative components
		dudz = np.sinh(z + h) * np.cos(x - t / epsilon) / np.sinh(h) + psi \
			 * kappa * np.sinh(kappa * (z + h)) * np.cos(kappa * x - tau * t \
			 / epsilon) / np.sinh(kappa * h)
		dwdz = np.cosh(z + h) * np.sin(x - t / epsilon) / np.sinh(h) + psi \
			 * kappa * np.cosh(kappa * (z + h)) * np.sin(kappa * x - tau * t \
			 / epsilon) / np.sinh(kappa * h)
		return np.array([dudz, dwdz])

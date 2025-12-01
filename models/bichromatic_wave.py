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
			The kinematic viscosity $\nu'$ of seawater.
		wavenum : ndarray
			1D array of `float` values, the wavenumbers $k'_1$ and $k'_2$,
			computed as $$k'_i = \frac{2 \pi}{\lambda'_i}.$$
		gravity : float
			The gravity **g** acting on the fluid, non-dimensionalized as,
			$$g = \frac{g'k'_1}{\omega^{\prime 2}_1}.$$
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
			$$\text{period} = \text{period}'\omega'_1.$$
		froude_num : float
			1D array of `float` values, the Froude numbers $Fr_1$ and $Fr_2$,
			computed as $$Fr_i = \frac{\omega'_i}{\sqrt{g'k'_i}}.$$
		reynolds_num : float
			1D array of `float` values, the Reynolds numbers $Re_1$ and $Re_2$
			of the wave, computed as $$Re_i = \frac{\omega'_i}{k^{\prime 2}_i 
			\nu'}.$$
		"""
		super().__init__(depth, amplitude, wavelength)
		self.gravity *= self.wavenum[0] / (self.angular_freq[0]
										* self.angular_freq[0])
		self.period *= self.angular_freq[0]

	def set_angular_freq(self):
		r"""
		Define the angular frequency omega with the dispersion relation,
		$$\omega'_i = \sqrt{g'k'_i \tanh(k'_i h')}.$$
		"""
		k, h = self.wavenum, self.depth
		self.angular_freq = np.sqrt(constants.g * k * np.tanh(k * h))

	def velocity(self, x, z, t):
		r"""
		Compute the fluid velocity $\boldsymbol{u} = \langle u, w \rangle,$
		$$u(x, z, t) = \epsilon \frac{\cosh{(z + h)}}{\sinh{(h)}} \cos{(x - t)}
					 + \psi\tau\frac{\cosh{(\kappa (z + h))}}{\sinh{(\kappa h)}}
					   \cos{(\kappa x - \tau t)},$$
		$$w(x, z, t) = \epsilon \frac{\sinh{(z + h)}}{\sinh{(h)}} \sin{(x - t)} 
					 + \psi\tau\frac{\sinh{(\kappa (z + h))}}{\sinh{(\kappa h)}}
					   \sin{(\kappa x - \tau t)},$$
		with, $$\epsilon = k'_1 A'_1, \quad \kappa = \frac{k'_2}{k'_1}, \quad
				\psi = k'_1 A'_2, \quad \tau = \frac{\omega'_2}{\omega'_1}.$$

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
		epsilon = self.steepness[0]
		kappa = self.wavenum[1] / self.wavenum[0]
		tau = self.angular_freq[1] / self.angular_freq[0]
		psi = self.wavenum[0] * self.amplitude[1]

		# velocity field components
		u = epsilon * np.cosh(z + h) * np.cos(x - t) / np.sinh(h) \
					+ psi * tau * np.cosh(kappa * (z + h)) \
					* np.cos(kappa * x - tau * t) / np.sinh(kappa * h)
		w = epsilon * np.sinh(z + h) * np.sin(x - t) / np.sinh(h) \
					+ psi * tau * np.sinh(kappa * (z + h)) \
					* np.sin(kappa * x - tau * t) / np.sinh(kappa * h)
		return np.array([u, w])

	def partial_t(self, x, z, t): 
		r"""
		Compute the partial derivative of the fluid with respect to time,
		$$\frac{\partial u}{\partial t} = \epsilon \frac{\cosh(z + h)}{\sinh(h)}
				\sin(x - t) + \psi \tau^2 \frac{\cosh(\kappa(z + h))}
				{\sinh(\kappa h)} \sin(\kappa x - \tau t),$$
		$$\frac{\partial w}{\partial t} = -\epsilon \frac{\sinh(z + h)}
				{\sinh(h)} \cos(x - t) - \psi \tau ^2
				\frac{\sinh(\kappa (z + h))}{\sinh(\kappa h)}
				\cos(\kappa x - \tau t).$$

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
		epsilon = self.steepness[0]
		kappa = self.wavenum[1] / self.wavenum[0]
		tau = self.angular_freq[1] / self.angular_freq[0]
		psi = self.wavenum[0] * self.amplitude[1]

		# partial derivative components
		dudt = epsilon * np.cosh(z + h) * np.sin(x - t) / np.sinh(h) \
					   + psi * tau * tau * np.cosh(kappa * (z + h)) \
					   * np.sin(kappa * x - tau * t) / np.sinh(kappa * h)
		dwdt = epsilon * np.sinh(z + h) * np.cos(x - t) / np.sinh(h) \
					   - psi * tau * tau * np.sinh(kappa * (z + h)) \
					   * np.cos(kappa * x - tau * t) / np.sinh(kappa * h)
		return np.array([dudt, -dwdt])

	def partial_x(self, x, z, t): 
		r"""
		Compute the partial derivative of the fluid with respect to the
		horizontal position,
		$$\frac{\partial u}{\partial x} = -\epsilon\frac{\cosh(z + h)}{\sinh(h)}
				\sin(x - t) - \psi \tau \kappa \frac{\cosh(\kappa (z + h))}
				{\sinh(\kappa h)} \sin (\kappa x - \tau t),$$
		$$\frac{\partial w}{\partial x} = \epsilon \frac{\sinh(z + h)}{\sinh(h)}
				\cos(x - t) + \psi \tau \kappa \frac{\sinh(\kappa (z + h))}
				{\sinh(\kappa h)} \cos(\kappa x - \tau t).$$

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
		epsilon = self.steepness[0]
		kappa = self.wavenum[1] / self.wavenum[0]
		tau = self.angular_freq[1] / self.angular_freq[0]
		psi = self.wavenum[0] * self.amplitude[1]

		# partial derivative components
		dudx = epsilon * np.cosh(z + h) * np.sin(x - t) / np.sinh(h) - psi \
					   * tau * kappa * np.cosh(kappa * (z + h)) \
					   * np.sin(kappa * x - tau * t) / np.sinh(kappa * h)
		dwdx = epsilon * np.sinh(z + h) * np.cos(x - t) / np.sinh(h) + psi \
					   * tau * kappa * np.sinh(kappa * (z + h)) \
					   * np.cos(kappa * x - tau * t) / np.sinh(kappa * h)
		return np.array([-dudx, dwdx])

	def partial_z(self, x, z, t):
		r"""
		Compute the partial derivative of the fluid with respect to the
		vertical position,
		$$\frac{\partial u}{\partial z} = \epsilon \frac{\sinh(z + h)}{\sinh(h)}
				\cos(x - t) + \psi \tau \kappa \frac{\sinh(\kappa (z + h))}
				{\sinh(\kappa h)} \cos(\kappa x - \tau t),$$
		$$\frac{\partial w}{\partial z} = \epsilon \frac{\cosh(z + h)}{\sinh(h)}
				\sin(x - t) + \psi \tau \kappa \frac{\cosh(\kappa (z + h))}
				{\sinh(\kappa h)} \sin(\kappa x - \tau t).$$

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
		epsilon = self.steepness[0]
		kappa = self.wavenum[1] / self.wavenum[0]
		tau = self.angular_freq[1] / self.angular_freq[0]
		psi = self.wavenum[0] * self.amplitude[1]

		# partial derivative components
		dudz = epsilon * np.sinh(z + h) * np.cos(x - t) / np.sinh(h) + psi \
					   * tau * kappa * np.sinh(kappa * (z + h)) \
					   * np.cos(kappa * x - tau * t) / np.sinh(kappa * h)
		dwdz = epsilon * np.cosh(z + h) * np.sin(x - t) / np.sinh(h) + psi \
					   * tau * kappa * np.cosh(kappa * (z + h)) \
					   * np.sin(kappa * x - tau * t) / np.sinh(kappa * h)
		return np.array([dudz, dwdz])

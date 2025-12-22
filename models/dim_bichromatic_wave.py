import numpy as np
from scipy import constants
from transport_framework import wave

class DimensionalBichromaticWave(wave.Wave):
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
		steepness : ndarray
			1D array of `float` values, the wave steepnesses
			$\epsilon = k'_1 A'_1$ and $k'_2 A'_2$.
		gravity : float
			The gravity **g** acting on the fluid.
		angular_freq : ndarray
			1D array of `float` values, the angular frequencies $\omega'_1$ and
			$\omega'_2$, computed using the dispersion relation,
			$$\omega'_i = \sqrt{g'k'_i \tanh(k'_i h')}.$$
		phase_velocity : ndarray
			 1D array of `float` values, the phase velocities $c'_1$ and $c'_2$,
			 computed as $$c'_i = \frac{\omega'_i}{k'_i}.$$
		period : ndarray
			1D array of `float` values, the periods of the wave, computed as
			$$\text{period}'_i = \frac{2\pi}{\omega'_i}.$$
		froude_num : float
			1D array of `float` values, the Froude numbers $Fr_1$ and $Fr_2$,
			computed as $$Fr_i = \frac{\omega'_i}{\sqrt{g'k'_i}}.$$
		reynolds_num : float
			1D array of `float` values, the Reynolds numbers $Re_1$ and $Re_2$
			of the wave, computed as $$Re_i = \frac{\omega'_i}{k^{\prime 2}_i 
			\nu'}.$$
		"""
		super().__init__(depth, amplitude, wavelength)

	def set_angular_freq(self):
		r"""
		Define the angular frequency omega with the dispersion relation,
		$$\omega'_i = \sqrt{g'k'_i \tanh(k'_i h')}.$$
		"""
		k, h = self.wavenum, self.depth
		self.angular_freq = np.sqrt(constants.g * k * np.tanh(k * h))

	def velocity(self, x, z, t):
		r"""
		Compute the fluid velocity $\boldsymbol{u}' = \langle u', w' \rangle,$
		$$u'(x', z', t') = \omega'_1 A'_1 \frac{\cosh{(k'_1(z' + h'))}}
						  {\sinh{(k'_1 h')}} \cos{(k'_1 x' - \omega'_1 t')} \
						 + \omega'_2 A'_2 \frac{\cosh{(k'_2 (z' + h'))}}
						  {\sinh{(k'_2 h')}} \cos{(k'_2 x' - \omega'_2 t')},$$
		$$w'(x', z', t') = \omega'_1 A'_1 \frac{\sinh{(k'_1(z' + h'))}}
						  {\sinh{(k'_1 h')}} \sin{(k'_1 x' - \omega'_1 t')} 
						 + \frac{\sinh{(k'_2 (z' + h'))}}{\sinh{(k'_2 h')}}
						   \sin{(k'_2 x - \omega'_2 t)}.$$

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
		h = self.depth
		k1, k2 = self.wavenum
		omega1, omega2 = self.angular_freq
		a1, a2 = self.amplitude

		# velocity field components
		u = a1 * omega1 * np.cosh(k1 * (z + h)) * np.cos(k1 * x - omega1 * t) \
			   / np.sinh(k1 * h) + a2 * omega2 * np.cosh(k2 * (z + h)) \
			   * np.cos(k2 * x - omega2 * t) / np.sinh(k2 * h)
		w = a1 * omega1 * np.sinh(k1 * (z + h)) * np.sin(k1 * x - omega1 * t) \
			   / np.sinh(k1 * h) + a2 * omega2 * np.sinh(k2 * (z + h)) \
			   * np.sin(k2 * x - omega2 * t) / np.sinh(k2 * h)
		return np.array([u, w])

	def partial_t(self, x, z, t): 
		r"""
		Compute the partial derivative of the fluid with respect to time,
		$$\frac{\partial u'}{\partial t'} = \omega^{\prime 2}_1 A'_1
				\frac{\cosh(k'_1(z' + h'))}{\sinh(k'_1 h')}
				\sin(k'_1 x' - \omega'_1 t') + \omega^{\prime 2}_2 A'_2
				\frac{\cosh(k'_2(z' + h'))}{\sinh(k'_2 h')}
				\sin(k'_2 x' - \omega'_2 t'),$$
		$$\frac{\partial w'}{\partial t'} = -\omega^{\prime 2}_1 A'_1
				\frac{\sinh(k'_1(z' + h'))}{\sinh(k'_1 h')}
				\cos(k'_1 x' - \omega'_1 t') - \omega^{\prime 2}_2 A'_2
				\frac{\sinh(k'_2 (z' + h'))}{\sinh(k'_2 h')}
				\cos(k'_2 x' - \omega'_2 t').$$

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
		h = self.depth
		k1, k2 = self.wavenum
		omega1, omega2 = self.angular_freq
		a1, a2 = self.amplitude

		# partial derivative components
		dudt = a1 * omega1 * omega1 * np.cosh(k1 * (z + h)) \
				  * np.sin(k1 * x - omega1 * t) / np.sinh(h) \
				  + a2 * omega2 * omega2 * np.cosh(k2 * (z + h)) \
				  * np.sin(k2 * x - omega2 * t) / np.sinh(k2 * h)
		dwdt = a1 * omega1 * omega1 * np.sinh(k1 * (z + h)) \
				  * np.cos(k1 * x - omega1 * t) / np.sinh(k1 * h) \
				  - a2 * omega2 * omega2 * np.sinh(k2 * (z + h)) \
				  * np.cos(k2 * x - omega2 * t) / np.sinh(k2 * h)
		return np.array([dudt, -dwdt])

	def partial_x(self, x, z, t): 
		r"""
		Compute the partial derivative of the fluid with respect to the
		horizontal position,
		$$\frac{\partial u'}{\partial x'} = -\omega'_1 k'_1 A'_1
				\frac{\cosh(k'_1(z' + h'))}{\sinh(k'_1 h')}
				\sin(k'_1 x' - \omega'_1 t') - \omega'_2 k'_2 A'_2
				\frac{\cosh(k'_2 (z' + h'))}{\sinh(k'_2 h')}
				\sin(k'_2 x' - \omega'_2 t'),$$
		$$\frac{\partial w'}{\partial x'} = \omega_1 k'_1 A'_1
				\frac{\sinh(k'_1(z' + h'))}{\sinh(k'_1 h')}
				\cos(k'_1 x' - \omega'_1 t') + \omega'_2 k'_2 A'_2
				\frac{\sinh(k'_2 (z' + h'))}{\sinh(k'_2 h')}
				\cos(k'_2 x' - \omega'_2 t').$$

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
		h = self.depth
		k1, k2 = self.wavenum
		omega1, omega2 = self.angular_freq
		a1, a2 = self.amplitude

		# partial derivative components
		dudx = omega1 * k1 * a1 * np.cosh(k1 * (z + h)) \
				  * np.sin(k1 * x - omega1 * t) / np.sinh(k1 * h) \
				  - a2 * k2 * omega2 * np.cosh(k2 * (z + h)) \
				  * np.sin(k2 * x - omega2 * t) / np.sinh(k2 * h)
		dwdx = omega1 * k1 * a1 * np.sinh(k1 * (z + h)) \
				  * np.cos(k1 * x - omega1 * t) / np.sinh(k1 * h) \
				  + a2 * k2 * omega2 * np.sinh(k2 * (z + h)) \
				  * np.cos(k2 * x - omega2 * t) / np.sinh(k2 * h)
		return np.array([-dudx, dwdx])

	def partial_z(self, x, z, t):
		r"""
		Compute the partial derivative of the fluid with respect to the
		vertical position,
		$$\frac{\partial u'}{\partial z'} = \omega'_1 k'_1 A'_1
				\frac{\sinh(k'_1(z' + h'))}{\sinh(k'_1 h')}
				\cos(k'_1 x' - \omega'_1 t') + \omega'_2 k'_2 A'_2
				\frac{\sinh(k'_2 (z' + h'))}{\sinh(k'_2 h')}
				\cos(k'_2 x' - \omega'_2 t'),$$
		$$\frac{\partial w'}{\partial z'} = \omega'_1 k'_1 A'_1
				\frac{\cosh(k'_1 (z' + h'))}{\sinh(k'_1 h')}
				\sin(k'_1 x' - \omega'_1 t') + \omega'_2 k'_2 A'_2
				\frac{\cosh(k'_2 (z' + h'))}
				{\sinh(k'_2 h')} \sin(k'_2 x' - \omega'_2 t').$$

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
		h = self.depth
		k1, k2 = self.wavenum
		omega1, omega2 = self.angular_freq
		a1, a2 = self.amplitude

		# partial derivative components
		dudz = omega1 * k1 * a1 * np.sinh(k1 * (z + h)) \
					  * np.cos(k1 * x - omega1 * t) / np.sinh(k1 * h) \
					  + omega2 * k2 * a2 * np.sinh(k2 * (z + h)) \
					   * np.cos(k2 * x - omega2 * t) / np.sinh(k2 * h)
		dwdz = omega1 * k1 * a1 * np.cosh(k1 * (z + h)) \
					  * np.sin(k1 * x - omega1 * t) / np.sinh(k1 * h) \
					  + omega2 * k2 * a2 * np.cosh(k2 * (z + h)) \
					   * np.sin(k2 * x - omega2 * t) / np.sinh(k2 * h)
		return np.array([dudz, dwdz])

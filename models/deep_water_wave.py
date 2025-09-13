import numpy as np
from scipy import constants
from transport_framework import wave

class DeepWaterWave(wave.Wave):
	"""Represent a dimensionless linear wave of infinitely deep water.[^1]"""

	def __init__(self, amplitude, wavelength, depth=50):
		r"""
		Attributes
		----------
		amplitude : float
			The amplitude of the wave *A'*.
		wavelength : float
			The wavelength *λ'*.
		depth : float, default=50
			The depth of the fluid *h'*.
		wavenum : float
			The wavenumber *k'*, computed as $$k' = \frac{2 \pi}{\lambda'}.$$
		kinematic_viscosity : float
			The kinematic viscosity ν' of seawater.
		gravity : ndarray
			The gravity **g** acting on the fluid, non-dimensionalized as,
			$$g' = \frac{g}{k'U^{\prime 2}}.$$
		angular_freq : float
			The angular frequency *ω'*, computed using the dispersion relation,
			$$\omega' = \sqrt{g'k'}.$$
		phase_velocity : float
			The phase velocity *c'*, computed as $$c' = \frac{\omega'}{k'}.$$
		period : float
			The period of the wave, computed as
			$$\text{period}' = \frac{2\pi}{\omega'}$$
			and non-dimensionalized as
			$$\text{period} = \text{period}' * k'U'.$$
		froude_num : float
			The Froude number *Fr*, computed as
			$$Fr = \sqrt{\frac{k'U'^2}{g'}}.$$
		reynolds_num : float
			The Reynolds number *Re* of the wave, computed as
			$$Re = \frac{U'}{k'ν'}.$$
		max_velocity : float
			The maximum velocity *U'* at the surface *z'* = 0, computed as
			$$U' = \omega' A'.$$

		References
		----------
		[^1]: [F. Santamaria et al. (2013).](
			  https://doi.org/10.1209/0295-5075/102/14003)
			  Stokes drift for inertial particles transported by water waves.
			  *EPL (Europhysics Letters)* 102(1), 14003.
		"""
		super().__init__(depth, amplitude, wavelength)
		self.gravity /= constants.g * self.froude_num ** 2
		self.max_velocity = self.angular_freq * self.amplitude
		self.period *= self.wavenum * self.max_velocity

	def set_angular_freq(self):
		r"""
		Define the angular frequency omega with the dispersion relation,
		$$\omega' = \sqrt{g'k'}.$$
		"""
		self.angular_freq = np.sqrt(constants.g * self.wavenum)

	def velocity(self, x, z, t):
		r"""
		Compute the fluid velocity, $$\textbf{u} = \langle u, w \rangle,$$
		$$u(x, z, t) = e^{z} \cos\Bigg(x - \frac{t}{\epsilon}\Bigg),$$
		$$w(x, z, t) = e^{z} \sin\Bigg(x - \frac{t}{\epsilon}\Bigg),$$
		where, $$\epsilon = Fr = k'A'.$$

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
		return np.array([np.exp(z) * np.cos(x - t / self.froude_num),
						 np.exp(z) * np.sin(x - t / self.froude_num)])

	def partial_t(self, x, z, t):
		r"""
		Compute the partial derivative of the fluid with respect to time,
		$$\frac{\partial \mathbf{u}}{\partial t} =
		\Bigg\langle \frac{w}{\epsilon}, \; -\frac{u}{\epsilon} \Bigg\rangle.$$

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
		u, w = self.velocity(x, z, t)
		return np.array([w / self.froude_num, -u / self.froude_num])

	def partial_x(self, x, z, t):
		r"""
		Compute the partial derivative of the fluid with respect to the
		horizontal position,
		$$\frac{\partial \mathbf{u}}{\partial x} = \langle -w, \; u \rangle.$$

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
		u, w = self.velocity(x, z, t)
		return np.array([-w, u])

	def partial_z(self, x, z, t):
		r"""
		Compute the partial derivative of the fluid with respect to the vertical
		position,
		$$\frac{\partial \mathbf{u}}{\partial z} = \langle u, \; w \rangle.$$

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
		return self.velocity(x, z, t)

	def material_derivative2(self, x, z, t):
		r"""
		Compute the second order material derivative, where
		$$\frac{\mathrm{D}^2\textbf{u}}{\mathrm{D}t^2} =
		\Bigg\langle e^{2z} - \frac{u}{\epsilon^2}, \quad
		w \Bigg(2 e^{2z} - \frac{1}{\epsilon^2}\Bigg) \Bigg\rangle.$$

		Parameters
		----------
		x, z : float or ndarray
			The horizontal and vertical position(s).
		t : float or ndarray
			The time(s) at which to evaluate the velocity.

		Returns
		-------
		ndarray
			1D array of `float` data, the vector components of the derivative.
		"""
		u, w = self.velocity(x, z, t)
		Fr = self.froude_num
		return np.array([np.exp(2 * z) / Fr - u / Fr ** 2,
						 w * (2 * np.exp(2 * z) - 1 / Fr ** 2)])

import numpy as np
from scipy import constants
from transport_framework import wave

class DeepLinearWave(wave.Wave):
	"""Represent a dimensionless linear wave of infinitely deep water.[^1]"""

	def __init__(self, amplitude, wavelength, depth=600):
		r"""
		Attributes
		----------
		amplitude : float
			The amplitude of the wave $A'$.
		wavelength : float
			The wavelength $\lambda'$.
		depth : float, default=600
			The depth of the fluid $h'$.
		wavenum : float
			The wavenumber, $$k' = 2 \pi / \lambda'.$$
		steepness : float
			The wave steepness, $$\epsilon = k'A'.$$
		kinematic_viscosity : float
			The kinematic viscosity $\nu'$ of seawater.
		gravity : ndarray
			The gravity $\boldsymbol{g}$ acting on the fluid,
			non-dimensionalized as,
			$$\boldsymbol{g} = \frac{\boldsymbol{g}'k'}{\omega^{\prime 2}}.$$
		angular_freq : float
			The angular frequency $\omega'$, computed using the dispersion
			relation, $$\omega' = \sqrt{g'k'}.$$
		phase_velocity : float
			The phase velocity, $$c' = \frac{\omega'}{k'}.$$
		period : float
			The period of the wave, $$\text{period}' = \frac{2\pi}{\omega'}$$
			non-dimensionalized as $$\text{period} = \text{period}'\omega'.$$
		froude_num : float
			The Froude number, $$Fr = \frac{\omega'}{\sqrt{g'k'}}.$$
		reynolds_num : float
			The Reynolds number of the wave, $$Re = \frac{\omega'}{k^{\prime 2}
			\nu'}.$$
		max_velocity : float
			The maximum velocity $U'$ at the surface $z' = 0$, computed as
			$$U' = \omega' A'.$$

		References
		----------
		[^1]: [F. Santamaria et al. (2013).](
			  https://doi.org/10.1209/0295-5075/102/14003)
			  Stokes drift for inertial particles transported by water waves.
			  *EPL (Europhysics Letters)* 102(1), 14003.
		"""
		super().__init__(depth, amplitude, wavelength)
		self.gravity *= self.wavenum / (self.angular_freq * self.angular_freq)
		self.max_velocity = self.angular_freq * amplitude
		self.period *= self.angular_freq

	def set_angular_freq(self):
		r"""
		Define the angular frequency omega with the dispersion relation,
		$$\omega' = \sqrt{g'k'}.$$
		"""
		self.angular_freq = np.sqrt(constants.g * self.wavenum)

	def velocity(self, x, z, t):
		r"""
		Compute the fluid velocity, $$\boldsymbol{u} = \langle u, w \rangle,$$
		$$u(x, z, t) = \epsilon e^{z} \cos(x - t),$$
		$$w(x, z, t) = \epsilon e^{z} \sin(x - t).$$

		Parameters
		----------
		x, z : float or ndarray
			The horizontal and vertical position(s).
		t : float or ndarray
			The time(s) at which to evaluate the velocity.

		Returns
		-------
		ndarray
			1D array of `float` data, the vector components $u$ and $w$.
		"""
		return np.array([self.steepness * np.exp(z) * np.cos(x - t),
						 self.steepness * np.exp(z) * np.sin(x - t)])

	def partial_t(self, x, z, t):
		r"""
		Compute the partial derivative of the fluid with respect to time,
		$$\frac{\partial \boldsymbol{u}}{\partial t} = \langle w, \; -u \rangle.$$

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
		return np.array([w, -u])

	def partial_x(self, x, z, t):
		r"""
		Compute the partial derivative of the fluid with respect to the
		horizontal position,
		$$\frac{\partial \boldsymbol{u}}{\partial x} = \langle -w, \; u \rangle.$$

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
		$$\frac{\partial \boldsymbol{u}}{\partial z} = \langle u, \; w \rangle.$$

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
		$$\frac{\mathrm{D}^2\boldsymbol{u}}{\mathrm{D}t^2} =
		\langle \epsilon^2 e^{2z} - u, w(2 \epsilon^2 e^{2z} - 1)\rangle.$$

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
		epsilon = self.steepness
		u, w = self.velocity(x, z, t)
		return np.array([epsilon * epsilon * np.exp(2 * z) - u,
						 w * (2 * epsilon * epsilon * np.exp(2 * z) - 1)])

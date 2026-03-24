import numpy as np
from scipy import constants
from transport_framework import wave

class DimensionalDeepLinearWave(wave.Wave):
	"""Represent a dimensional linear wave of infinitely deep water.[^1]"""

	def __init__(self, amplitude, wavelength, depth=600):
		r"""
		Attributes
		----------
		depth : float, default=600
			The depth of the fluid $h'$.
		amplitude : float
			The amplitude of the wave $A'$.
		wavelength : float
			The wavelength $\lambda'$.
		kinematic_viscosity : float
			The kinematic viscosity $\nu'$ of seawater.
		wavenum : float
			The wavenumber, $$k' = 2 \pi / \lambda'.$$
		steepness : float
			The wave steepness, $$\epsilon = k'A'.$$
		gravity : ndarray
			1D array of `float` data, the gravity $\boldsymbol{g'}$ acting on
			the fluid.
		angular_freq : float
			The angular frequency $\omega'$, computed using the dispersion
			relation, $$\omega' = \sqrt{g'k'}.$$
		phase_velocity : float
			The phase velocity, $$c' = \omega' / k'.$$
		period : float
			The period of the wave, where
			$$\text{period}' = 2\pi / \omega'.$$
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
		self.max_velocity = self.angular_freq * amplitude

	def set_angular_freq(self):
		r"""
		Define the angular frequency omega with the dispersion relation,
		$$\omega' = \sqrt{g'k'}.$$
		"""
		self.angular_freq = np.sqrt(constants.g * self.wavenum)

	def velocity(self, x, z, t):
		r"""
		Compute the fluid velocity, $$\boldsymbol{u}' = \langle u', w' \rangle,$$
		$$u'(x', z', t') = U'e^{k'z'} \cos(k'x' - \omega' t'),$$
		$$w'(x', z', t') = U'e^{k'z'} \sin(k'x' - \omega' t').$$

		Parameters
		----------
		x : float or ndarray
			The horizontal position(s) at which to evaluate the velocity.
		z : float or ndarray
			The vertical position(s) at which to evaluate the velocity.
		t : float or ndarray
			The time(s) at which to evaluate the velocity.

		Returns
		-------
		ndarray
			1D array of `float` data, the vector components $u$ and $w$.
		"""
		u = self.max_velocity
		k = self.wavenum
		omega = self.angular_freq

		return np.array([u * np.exp(k * z) * np.cos(k * x - omega * t), 
						 u * np.exp(k * z) * np.sin(k * x - omega * t)])

	def partial_t(self, x, z, t): 
		r"""
		Compute the partial derivative of the fluid with respect to time,
		$$\frac{\partial \boldsymbol{u'}}{\partial t'} =
		\langle \omega' w', \; -\omega' u' \rangle.$$

		Parameters
		----------
		x : float or ndarray
			The horizontal position(s) at which to evaluate the derivative.
		z : float or ndarray
			The vertical position(s) at which to evaluate the derivative.
		t : float or ndarray
			The time(s) at which to evaluate the derivative.

		Returns
		-------
		ndarray
			1D array of `float` data, the vector components of the derivative.
		"""
		omega = self.angular_freq
		u, w = self.velocity(x, z, t)
		return np.array([omega * w, omega * -u])

	def partial_x(self, x, z, t): 
		r"""
		Compute the partial derivative of the fluid with respect to the
		horizontal position, $$\frac{\partial \boldsymbol{u}'}{\partial x'}
								= \langle -k'w', \; k'u' \rangle.$$

		Parameters
		----------
		x : float or ndarray
			The horizontal position(s) at which to evaluate the derivative.
		z : float or ndarray
			The vertical position(s) at which to evaluate the derivative.
		t : float or ndarray
			The time(s) at which to evaluate the derivative.

		Returns
		-------
		ndarray
			1D array of `float` data, the vector components of the derivative.
		"""
		k = self.wavenum
		u, w = self.velocity(x, z, t)
		return np.array([k * -w, k * u])

	def partial_z(self, x, z, t):
		r"""
		Compute the partial derivative of the fluid with respect to the vertical
		position, $$\frac{\partial \boldsymbol{u}'}{\partial z'}
					= \langle k'u', \; k'w' \rangle.$$

		Parameters
		----------
		x : float or ndarray
			The horizontal position(s) at which to evaluate the derivative.
		z : float or ndarray
			The vertical position(s) at which to evaluate the derivative.
		t : float or ndarray
			The time(s) at which to evaluate the derivative.

		Returns
		-------
		ndarray
			1D array of `float` data, the vector components of the derivative.
		"""
		return self.wavenum * self.velocity(x, z, t)

	def material_derivative2(self, x, z, t):
		r"""
		Compute the second order material derivative, where
		$$\frac{\mathrm{D}^2\boldsymbol{u}'}{\mathrm{D}t'^2} =
		\langle U'^2 e^{2k'z'} \omega' k' - \omega'^2 u', \quad
		w'(2 e^{2k'z'} U'^2 k'^2 - \omega'^2) \rangle.$$

		Parameters
		----------
		x : float or ndarray
			The x position(s) at which to evaluate the fluid velocity.
		z : float or ndarray
			The z position(s) at which to evaluate the velocity and derivative.
		t : float or ndarray
			The time(s) at which to evaluate the velocity.

		Returns
		-------
		ndarray
			1D array of `float` data, the vector components of the derivative.
		"""
		k = self.wavenum
		epsilon = self.steepness
		omega = self.angular_freq
		u, w = self.velocity(x, z, t)

		return np.array([k * w * self.max_velocity ** 2 * np.exp(2 * k * z)
						   - omega ** 2 * u, w * (2 * np.exp(2 * k * z)
						   * (self.max_velocity * k) ** 2 - omega ** 2)])

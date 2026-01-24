import numpy as np
from scipy import constants

from transport_framework import wave
from utils.colors import print_warning

class BichromaticWave(wave.Wave):
	"""Represent a dimensionless bichromatic wave of arbitrarily deep water."""

	def __init__(self, depth, amplitude, wavelength, slope=0,
				 include_subharmonics=True):
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
		slope : float, default=0
			The slope of the seabed.
		include_subharmonics : bool, defualt=True
			Whether to include subharmonic effects.
		kinematic_viscosity : float
			The kinematic viscosity $\nu'$ of seawater.
		wavenum : ndarray
			1D array of `float` values, the wavenumbers $k'_1$ and $k'_2$,
			computed as $$k'_i = \frac{2 \pi}{\lambda'_i}.$$
		steepness : ndarray
			1D array of `float` values, the wave steepnesses
			$\epsilon = k'_1 A'_1$ and $k'_2 A'_2$.
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
			$$\text{period}_i = \text{period}_i'\omega_i'.$$
		froude_num : float
			1D array of `float` values, the Froude numbers $Fr_1$ and $Fr_2$,
			computed as $$Fr_i = \frac{\omega'_i}{\sqrt{g'k'_i}}.$$
		reynolds_num : float
			1D array of `float` values, the Reynolds numbers $Re_1$ and $Re_2$
			of the wave, computed as $$Re_i = \frac{\omega'_i}{k^{\prime 2}_i 
			\nu'}.$$
		"""
		super().__init__(depth, amplitude, wavelength)
		self.slope = slope
		self.include_subharmonics = include_subharmonics
		self.gravity *= self.wavenum[0] / (self.angular_freq[0]
										* self.angular_freq[0])
		self.period *= self.angular_freq
		# warn if the |h_x| << hk_g condition is not met
		if np.abs(self.wavenum[0] - self.wavenum[1]) * depth / 10 \
			< np.abs(slope): print_warning('Slope is too steep.')

	def set_angular_freq(self):
		r"""
		Define the angular frequency omega with the dispersion relation,
		$$\omega'_i = \sqrt{g'k'_i \tanh(k'_i h')}.$$
		"""
		k, h = self.wavenum, self.depth
		self.angular_freq = np.sqrt(constants.g * k * np.tanh(k * h))

	def velocity(self, x, z, t):
		r"""
		Compute the first order fluid velocity,
		$\boldsymbol{u}^{(1)} = \langle u^{(1)}, w^{(1)} \rangle,$
		$$u^{(1)}(x, z, t) = \epsilon \frac{\cosh{(z + h)}}{\sinh{(h)}}
							 \cos{(x - t)} + \psi\tau\frac{\cosh{(\kappa (z
						   + h))}}{\sinh{(\kappa h)}}
							 \cos{(\kappa x - \tau t)},$$
		$$w^{(1)}(x, z, t) = \epsilon \frac{\sinh{(z + h)}}{\sinh{(h)}}
							 \sin{(x - t)} + \psi\tau
							 \frac{\sinh{(\kappa (z + h))}}{\sinh{(\kappa h)}}
							 \sin{(\kappa x - \tau t)},$$
		with, $$\epsilon = k'_1 A'_1, \quad \kappa = \frac{k'_2}{k'_1}, \quad
				\psi = k'_1 A'_2, \quad \tau = \frac{\omega'_2}{\omega'_1},$$
		Optionally, the second order horizontal component,
		$$u^{(2)}(x, t) = -K_u \psi \epsilon \frac{c_g^2 + gh - \frac{1}{2}
						  cc_g}{hc(h - c^2_g/g)} \cos(\upsilon x - \zeta t 
						+ \phi_u),$$
		where $$\upsilon = \frac{k'_g}{k'_1}, \quad
				\zeta = \frac{\omega'_g}{\omega'_1},$$
		is added to include subharmonic effects, thus returning
		$\boldsymbol{u} = \langle u^{(1)} + u^{(2)}, w^{(1)} \rangle.$

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
		# dimensional parameters
		k1, k2 = self.wavenum
		omega1, omega2 = self.angular_freq
		c1 = self.phase_velocity[0]

		# dimensionless parameters
		h = k1 * self.seabed(x)
		epsilon = self.steepness[0]
		kappa = k2 / k1
		tau = omega2 / omega1
		psi = k1 * self.amplitude[1]
		upsilon = np.abs(k1 - k2) / k1
		zeta = np.abs(omega1 - omega2) / omega1
		cg = np.abs(omega1 - omega2) / np.abs(k1 - k2) / c1
		c = np.average(self.phase_velocity) / c1
		g = -self.gravity[1]

		# velocity field components
		u = epsilon * np.cosh(z + h) * np.cos(x - t) / np.sinh(h) \
					+ psi * tau * np.cosh(kappa * (z + h)) \
					* np.cos(kappa * x - tau * t) / np.sinh(kappa * h)
		w = epsilon * np.sinh(z + h) * np.sin(x - t) / np.sinh(h) \
					+ psi * tau * np.sinh(kappa * (z + h)) \
					* np.sin(kappa * x - tau * t) / np.sinh(kappa * h)
		if self.include_subharmonics:
			return np.array([u + self.subharmonic(x, t), w])
		else:
			return np.array([u, w])

	def partial_t(self, x, z, t): 
		r"""
		Compute the partial derivative of the fluid with respect to time,
		$$\frac{\partial u}{\partial t} = \epsilon \frac{\cosh(z + h)}{\sinh(h)}
				\sin(x - t) + \psi \tau^2
				\frac{\cosh(\kappa(z + h))}{\sinh(\kappa h)}
				\sin(\kappa x - \tau t)\\- K_u \psi \epsilon
				\zeta \frac{c_g^2 + gh - \frac{1}{2} cc_g}{hc(h - c^2_g/g)}
				\sin(\upsilon x - \zeta t + \phi_u),$$
		$$\frac{\partial w}{\partial t} = -\epsilon \frac{\sinh(z + h)}
				{\sinh(h)} \cos(x - t) - \psi \tau ^2
				\frac{\sinh(\kappa (z + h))}{\sinh(\kappa h)}
				\cos(\kappa x - \tau t).$$
		If subharmonic effects are neglected, we set $K_u = 0.$

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
		# dimensional parameters
		k1, k2 = self.wavenum
		omega1, omega2 = self.angular_freq
		c1 = self.phase_velocity[0]

		# dimensionless parameters
		h = k1 * self.seabed(x)
		epsilon = self.steepness[0]
		kappa = k2 / k1
		tau = omega2 / omega1
		psi = k1 * self.amplitude[1]
		k_u = self.k_u(x) if self.include_subharmonics else 0
		upsilon = np.abs(k1 - k2) / k1
		zeta = np.abs(omega1 - omega2) / omega1
		cg = np.abs(omega1 - omega2) / np.abs(k1 - k2) / c1
		c = np.average(self.phase_velocity) / c1
		g = -self.gravity[1]

		# partial derivative components
		dudt = epsilon * np.cosh(z + h) * np.sin(x - t) / np.sinh(h) \
					   + psi * tau * tau * np.cosh(kappa * (z + h)) \
					   * np.sin(kappa * x - tau * t) / np.sinh(kappa * h) \
					   - k_u * psi * epsilon * zeta * (cg * cg + g * h
					   - 0.5 * c * cg) / (h * c * (h - cg * cg / g)) \
					   * np.sin(upsilon * x - zeta * t - self.phi_u(x))
		dwdt = epsilon * np.sinh(z + h) * np.cos(x - t) / np.sinh(h) \
					   - psi * tau * tau * np.sinh(kappa * (z + h)) \
					   * np.cos(kappa * x - tau * t) / np.sinh(kappa * h)
		return np.array([dudt, -dwdt])

	def partial_x(self, x, z, t): 
		r"""
		Compute the partial derivative of the fluid with respect to the
		horizontal position,
		$$\frac{\partial u}{\partial x} = -\epsilon\frac{\cosh(z + h)}{\sinh(h)}
				\sin(x - t) - \psi \tau \kappa
				\frac{\cosh(\kappa (z + h))}{\sinh(\kappa h)}
				\sin(\kappa x - \tau t)\\- K_u \psi \epsilon
				\upsilon \frac{c_g^2 + gh - \frac{1}{2} cc_g}{hc(h - c^2_g/g)}
				\sin(\upsilon x - \zeta t + \phi_u),$$
		$$\frac{\partial w}{\partial x} = \epsilon \frac{\sinh(z + h)}{\sinh(h)}
				\cos(x - t) + \psi \tau \kappa \frac{\sinh(\kappa (z + h))}
				{\sinh(\kappa h)} \cos(\kappa x - \tau t).$$
		If subharmonic effects are neglected, we set $K_u = 0.$

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
		# dimensional parameters
		k1, k2 = self.wavenum
		omega1, omega2 = self.angular_freq
		c1 = self.phase_velocity[0]

		# dimensionless parameters
		h = k1 * self.seabed(x)
		epsilon = self.steepness[0]
		kappa = k2 / k1
		tau = omega2 / omega1
		psi = k1 * self.amplitude[1]
		k_u = self.k_u(x) if self.include_subharmonics else 0
		upsilon = np.abs(k1 - k2) / k1
		zeta = np.abs(omega1 - omega2) / omega1
		cg = np.abs(omega1 - omega2) / np.abs(k1 - k2) / c1
		c = np.average(self.phase_velocity) / c1
		g = -self.gravity[1]

		# partial derivative components
		dudx = epsilon * np.cosh(z + h) * np.sin(x - t) / np.sinh(h) - psi \
					   * tau * kappa * np.cosh(kappa * (z + h)) \
					   * np.sin(kappa * x - tau * t) / np.sinh(kappa * h) \
					   - k_u * psi * epsilon * upsilon * (cg * cg + g
					   * h - 0.5 * c * cg) / (h * c * (h - cg * cg / g)) \
					   * np.sin(upsilon * x - zeta * t - self.phi_u(x))
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
		h = self.wavenum[0] * self.seabed(x)
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

	def subharmonic(self, x, t):
		r"""
		Return the subharmonic component: the second order horizontal velocity,
		$$u^{(2)}(x, t) = -K_u \psi \epsilon \frac{c_g^2 + gh - \frac{1}{2}
						  cc_g}{hc(h - c^2_g/g)} \cos(\upsilon x - \zeta t 
						+ \phi_u).$$

		Parameters
		----------
		x : float or ndarray
			The horizontal position(s).
		t : float or ndarray
			The time(s) at which to evaluate the second order velocity.

		Returns
		-------
		float or ndarray
			The second order horizontal velocity.
		"""
		# dimensional parameters
		k1, k2 = self.wavenum
		omega1, omega2 = self.angular_freq
		c1 = self.phase_velocity[0]

		# dimensionless parameters
		h = k1 * self.seabed(x)
		epsilon = self.steepness[0]
		psi = k1 * self.amplitude[1]
		upsilon = np.abs(k1 - k2) / k1
		zeta = np.abs(omega1 - omega2) / omega1
		cg = np.abs(omega1 - omega2) / np.abs(k1 - k2) / c1
		c = np.average(self.phase_velocity) / c1
		g = -self.gravity[1]

		# horizontal second order velocity (subharmonic component)
		return -self.k_u(x) * psi * epsilon * (cg * cg + g * h - 0.5 * c * cg) \
			   / (h * c * (h - cg * cg / g)) * np.cos(upsilon * x - zeta * t
			   - self.phi_u(x))

	def seabed(self, x=0):
		"""Return the depth of the seabed at horizontal position `x`."""
		return self.slope * x / self.wavenum[0] + self.depth

	def beta(self, x):
		r"""Return $$\beta \equiv \frac{|h'_{x'}|}{k'_g h'(x')}.$$"""
		kg = np.abs(self.wavenum[0] - self.wavenum[1])
		beta = np.abs(self.slope) / (kg * self.seabed(x))
		if np.any(beta) > 0.5: print_warning('Coefficient \u03B2 is outside '
										   + 'the pre-computed range.')
		return beta

	def xi(self, x):
		r"""Return $$\xi = h'(x') \frac{k'_1 + k'_2}{2}.$$"""
		k = (self.wavenum[0] + self.wavenum[1]) / 2
		xi = k * self.seabed(x)
		if np.any(xi) > 4.12: print_warning('Coefficient \u03BE is outside the '
										  + 'pre-computed range.')
		return xi

	def k_u(self, x):
		r"""Return $$K_u(\xi, \beta) = \tanh(p_7 \xi^{p_8} \beta^{p_9}).$$"""
		p7 = 0.3917
		p8 = 0.9522
		p9 = -0.4982
		return np.tanh(p7 * self.xi(x) ** p8 * self.beta(x) ** p9)

	def phi_u(self, x):
		r"""Return $$\phi_u(\xi, \beta) = p_{10}\xi^{p_{11}}\beta^{p_{12}}.$$"""
		p10 = 0.8919
		p11 = -1.3034
		p12 = 0.5583
		return p10 * self.xi(x) ** p11 * self.beta(x) ** p12

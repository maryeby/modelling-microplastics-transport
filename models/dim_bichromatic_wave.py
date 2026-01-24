import numpy as np
from scipy import constants
from transport_framework import wave
from utils.colors import print_warning

class DimensionalBichromaticWave(wave.Wave):
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
		self.slope = slope
		self.include_subharmonics = include_subharmonics
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
		Compute the first order fluid velocity, $\boldsymbol{u}^{(1)\prime} 
		= \langle u^{(1)\prime}, w^{(1)\prime} \rangle,$
		$$u^{(1)\prime}(x', z', t') = \omega'_1 A'_1
				\frac{\cosh{(k'_1(z' + h'))}}{\sinh{(k'_1 h')}}
				\cos{(k'_1 x' - \omega'_1 t')} + \omega'_2 A'_2
				\frac{\cosh{(k'_2 (z' + h'))}}{\sinh{(k'_2 h')}}
				\cos{(k'_2 x' - \omega'_2 t')},$$
		$$w^{(1)\prime}(x', z', t') = \omega'_1 A'_1
				\frac{\sinh{(k'_1(z' + h'))}}{\sinh{(k'_1 h')}}
				\sin{(k'_1 x' - \omega'_1 t')} + \omega'_2 A'_2
				\frac{\sinh{(k'_2 (z' + h'))}}{\sinh{(k'_2 h')}}
				\sin{(k'_2 x - \omega'_2 t)}.$$
		Optionally, the second order horizontal component,
		$$u^{(2)\prime} = -K_u \frac{A'_1 A'_2}{h^{\prime 2} c'} \cdot
						  \frac{c^{\prime 2}_g + g'h' - \frac{1}{2} c'c'_g}{1
						- \frac{c^{\prime 2}_g}{g'h'}} \cos{(k'_g x'
						- \omega'_g t' + \phi_u)},$$
		is added to include subharmonic effects, thus returning $\boldsymbol{u}'
		= \langle u^{(1)\prime} + u^{(2)\prime}, w^{(1)\prime} \rangle.$

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
		h = self.seabed(x)
		k1, k2 = self.wavenum
		a1, a2 = self.amplitude
		omega1, omega2 = self.angular_freq

		# velocity field components
		u = a1 * omega1 * np.cosh(k1 * (z + h)) * np.cos(k1 * x - omega1 * t) \
			   / np.sinh(k1 * h) + a2 * omega2 * np.cosh(k2 * (z + h)) \
			   * np.cos(k2 * x - omega2 * t) / np.sinh(k2 * h)
		w = a1 * omega1 * np.sinh(k1 * (z + h)) * np.sin(k1 * x - omega1 * t) \
			   / np.sinh(k1 * h) + a2 * omega2 * np.sinh(k2 * (z + h)) \
			   * np.sin(k2 * x - omega2 * t) / np.sinh(k2 * h)
		if self.include_subharmonics:
			return np.array([u + self.subharmonic(x, t), w])
		else:
			return np.array([u, w])

	def partial_t(self, x, z, t): 
		r"""
		Compute the partial derivative of the fluid with respect to time,
		$$\frac{\partial u'}{\partial t'} = \omega^{\prime 2}_1 A'_1
				\frac{\cosh(k'_1(z' + h'))}{\sinh(k'_1 h')}
				\sin(k'_1 x' - \omega'_1 t') + \omega^{\prime 2}_2 A'_2
				\frac{\cosh(k'_2(z' + h'))}{\sinh(k'_2 h')}
				\sin(k'_2 x' - \omega'_2 t')\\-K_u \omega'_g
				\frac{A'_1 A'_2}{h^{\prime 2} c'} \cdot \frac{(c_g^{\prime 2}
				- c'c'_g / 2 + g'h')}{(1 - c_g^{\prime 2}/(g'h'))}
				\sin(x' k'_g - t' \omega'_g + \phi_u),$$
		$$\frac{\partial w'}{\partial t'} = -\omega^{\prime 2}_1 A'_1
				\frac{\sinh(k'_1(z' + h'))}{\sinh(k'_1 h')}
				\cos(k'_1 x' - \omega'_1 t') - \omega^{\prime 2}_2 A'_2
				\frac{\sinh(k'_2 (z' + h'))}{\sinh(k'_2 h')}
				\cos(k'_2 x' - \omega'_2 t').$$
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
		h = self.seabed(x)
		k1, k2 = self.wavenum
		a1, a2 = self.amplitude
		omega1, omega2 = self.angular_freq
		k_u = self.k_u(x) if self.include_subharmonics else 0
		k = (k1 + k2) / 2
		c = np.sqrt(constants.g * h) * np.sqrt(np.tanh(k * h) / (k * h))
		omega_g = np.abs(omega1 - omega2)
		kg = np.abs(k1 - k2)
		cg = omega_g / kg

		# partial derivative components
		dudt = a1 * omega1 * omega1 * np.cosh(k1 * (z + h)) \
				  * np.sin(k1 * x - omega1 * t) / np.sinh(h) \
				  + a2 * omega2 * omega2 * np.cosh(k2 * (z + h)) \
				  * np.sin(k2 * x - omega2 * t) / np.sinh(k2 * h) - k_u \
				  * omega_g * a1 * a2 / (h * h * c) * (cg * cg - c * cg / 2
				  + constants.g * h) / (1 - cg * cg / (constants.g * h)) \
				  * np.sin(kg * x - omega_g * t + self.phi_u(x))
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
				\sin(k'_2 x' - \omega'_2 t')\\+ K_u k'_g
				\frac{A'_1 A'_2}{h^{\prime 2} c'} \cdot \frac{c_g^{\prime 2}
			  - c'c'_g / 2 + g'h'}{1 - c_g^{\prime 2} / (g'h')} \sin(x' k'_g
			  - \omega'_g t' + \phi_u),$$
		$$\frac{\partial w'}{\partial x'} = \omega_1 k'_1 A'_1
				\frac{\sinh(k'_1(z' + h'))}{\sinh(k'_1 h')}
				\cos(k'_1 x' - \omega'_1 t') + \omega'_2 k'_2 A'_2
				\frac{\sinh(k'_2 (z' + h'))}{\sinh(k'_2 h')}
				\cos(k'_2 x' - \omega'_2 t').$$
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
		h = self.seabed(x)
		k1, k2 = self.wavenum
		a1, a2 = self.amplitude
		omega1, omega2 = self.angular_freq
		k_u = self.k_u(x) if self.include_subharmonics else 0
		k = (k1 + k2) / 2
		c = np.sqrt(constants.g * h) * np.sqrt(np.tanh(k * h) / (k * h))
		omega_g = np.abs(omega1 - omega2)
		kg = np.abs(k1 - k2)
		cg = omega_g / kg

		# partial derivative components
		dudx = omega1 * k1 * a1 * np.cosh(k1 * (z + h)) \
				  * np.sin(k1 * x - omega1 * t) / np.sinh(k1 * h) \
				  - a2 * k2 * omega2 * np.cosh(k2 * (z + h)) \
				  * np.sin(k2 * x - omega2 * t) / np.sinh(k2 * h) + k_u \
				  * kg * a1 * a2 / (h * h * c) * (cg * cg - c * cg / 2
				  + constants.g * h) / (1 - cg * cg / (constants.g * h)) \
				  * np.sin(kg * x - omega_g * t + self.phi_u(x))
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
		h = self.seabed(x)
		k1, k2 = self.wavenum
		a1, a2 = self.amplitude
		omega1, omega2 = self.angular_freq

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

	def subharmonic(self, x, t):
		r"""
		Return the subharmonic component: the second order horizontal velocity,
		$$u^{(2)\prime} = -K_u \frac{A'_1 A'_2}{h^{\prime 2} c'} \cdot
						  \frac{c^{\prime 2}_g + g'h' - \frac{1}{2} c'c'_g}{1
						- \frac{c^{\prime 2}_g}{g'h'}} \cos{(k'_g x'
						- \omega'_g t' + \phi_u)}.$$

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
		h = self.seabed(x)
		k1, k2 = self.wavenum
		a1, a2 = self.amplitude
		omega1, omega2 = self.angular_freq
		k = (k1 + k2) / 2
		c = np.sqrt(constants.g * h) * np.sqrt(np.tanh(k * h) / (k * h))
		omega_g = np.abs(omega1 - omega2)
		kg = np.abs(k1 - k2)
		cg = omega_g / kg

		# horizontal second order velocity (subharmonic component)
		return -self.k_u(x) * a1 * a2 / (h * h * c) * (cg * cg + constants.g * h
			   - 0.5 * c * cg) / (1 - cg * cg / (constants.g * h)) * np.cos(kg
			   * x - omega_g * t + self.phi_u(x))

	def seabed(self, x=0):
		"""Return the depth of the seabed at horizontal position `x`."""
		return self.slope * x + self.depth

	def beta(self, x):
		r"""Return $$\beta \equiv \frac{|h'_{x'}|}{k'_g h'(x')}.$$"""
		beta = np.abs(self.slope) / (np.abs(self.wavenum[0] - self.wavenum[1])
								  * self.seabed(x))
		if np.any(beta) > 0.5: print_warning('Coefficient \u03B2 is outside '
										   + 'the pre-computed range.')
		return beta

	def xi(self, x):
		r"""Return $$\xi = h'(x') \frac{k'_1 + k'_2}{2}.$$"""
		xi = (self.wavenum[0] + self.wavenum[1]) / 2 * self.seabed(x)
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

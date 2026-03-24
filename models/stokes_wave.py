import numpy as np
from scipy import constants
from transport_framework import wave

class StokesWave(wave.Wave):
	"""Represent a dimensionless, arbitrarily deep 5th-order Stokes wave.[^1]"""

	def __init__(self, depth, amplitude, wavelength):
		r"""
		Attributes
		----------
		depth : float
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
		gravity : float
			The gravity $g$ acting on the fluid, non-dimensionalized as,
			$$g = \frac{g'k'}{\omega^{\prime 2}}.$$
		angular_freq : float
			The angular frequency $\omega'$, computed using the dispersion
			relation, $$\omega' = \sqrt{g'k' \tanh(k'h')}.$$
		phase_velocity : float
			The phase velocity, $$c' = \omega' / k'.$$
		period : float
			The period of the wave,
			$$\text{period}' = \frac{2\pi}{\omega'},$$
			and non-dimensionalized as
			$$\text{period} = \text{period}'\omega'.$$
		froude_num : float
			The Froude number $$Fr = \frac{\omega'}{\sqrt{g'k'}}.$$
		reynolds_num : float
			The Reynolds number of the wave, $$Re = \frac{\omega'}{k^{\prime 2}
			\nu'}.$$
		order : int
			The order of the Stokes wave solution, 5 for fifth-order.
		a : ndarray
			An array of `float` coefficients $A_{ij}$ from Table 1 in [1].
		c0, c2, c4 : float
			Coefficients for the 5th order solution from Table 1 in [1].
		mean_speed : float
			The mean horizontal fluid speed $\bar{u}$ as defined by (13) in [1].

		References
		----------
		[^1]: [J.D. Fenton (1985).](https://doi.org/10.1061/(ASCE)0733-950X
								   (1985)111:2(216))
			  A Fifth‐Order Stokes Theory for Steady Waves.
			  *Journal of Waterway, Port, Coastal, and Ocean Engineering*
			  111(2), 216–234.
		"""
		super().__init__(depth, amplitude, wavelength)
		self.gravity *= self.wavenum / (self.angular_freq * self.angular_freq)
		self.period *= self.angular_freq
		self.order = 5
		self.set_coefficients()
		self.mean_speed = (self.c0 + self.steepness * self.steepness * self.c2
								   + self.steepness ** 4 * self.c4) / self.c0

	def set_angular_freq(self):
		r"""
		Define the angular frequency omega with the dispersion relation,
		$$\omega' = \sqrt{g'k' \tanh(k'h')}.$$
		"""
		k, h = self.wavenum, self.depth
		self.angular_freq = np.sqrt(constants.g * k * np.tanh(k * h))

	def set_coefficients(self):
		"""Define the coefficients for the fifth-order solution."""
		k, h = self.wavenum, self.depth
		s = 1 / np.cosh(2 * k * h)
		self.a = np.zeros((self.order + 1, self.order + 1))
		self.a[1, 1] = 1 / np.sinh(k * h)
		self.a[2, 2] = 3 * s * s / (2 * (1 - s) * (1 - s))
		self.a[3, 1] = (-4 - 20 * s + 10 * s * s - 13 * s ** 3) \
						/ (8 * np.sinh(k * h) * (1 - s) ** 3)
		self.a[3, 3] = (-2 * s * s + 11 * s ** 3) / (8 * np.sinh(k * h) \
						* (1 - s) ** 3)
		self.a[4, 2] = (12 * s - 14 * s * s - 264 * s ** 3 - 45 * s ** 4
						- 13 * s ** 5) / (24 * (1 - s) ** 5)
		self.a[4, 4] = (10 * s ** 3 - 174 * s ** 4 + 291 * s ** 5 + 278 * s
						** 6) / (48 * (3 + 2 * s) * (1 - s) ** 5)
		self.a[5, 1] = (-1184 + 32 * s + 13232 * s * s + 21712 * s ** 3
						+ 20940 * s ** 4 + 12554 * s ** 5 - 500 * s ** 6
						- 3341 * s ** 7 - 670 * s ** 8) / (64 * np.sinh(k * h)
						* (3 + 2 * s) * (4 + s) * (1 - s) ** 6)
		self.a[5, 3] = (4 * s + 105 * s * s + 198 * s ** 3 - 1376 * s ** 4
						- 1302 * s ** 5 - 117 * s ** 6 + 58 * s ** 7) \
						/ (32 * np.sinh(k * h) * (3 + 2 * s) * (1 - s) ** 6)
		self.a[5, 5] = (-6 * s ** 3 + 272 * s ** 4 - 1552 * s ** 5 + 852 * s \
						** 6 + 2029 * s ** 7 + 430 * s ** 8) / (64 * np.sinh(k
						* h) * (3 + 2 * s) * (4 + s) * (1 - s) ** 6)
		self.c0 = np.sqrt(np.tanh(k * h))
		self.c2 = np.sqrt(np.tanh(k * h)) * (2 + 7 * s ** 2) \
					/ (4 * (1 - s) ** 2)
		self.c4 = np.sqrt(np.tanh(k * h)) * (4 + 32 * s - 116 * s * s - 400 * s
					** 3 - 71 * s ** 4 + 146 * s ** 5) / (32 * (1 - s) ** 5)

	def velocity(self, x, z, t):
		r"""
		Compute the fluid velocity $\boldsymbol{u} = \langle u, w \rangle,$
		$$u(x, z, t) = 1 - \bar{u} + \sum_{i=1}^{5} \epsilon^i
			\sum_{j=1}^i jA_{ij} \cosh{(jz)} \cos{(j(x - t))},$$
		$$w(x, z, t) = \sum_{i=1}^{5} \epsilon^i \sum_{j=1}^i jA_{ij} 
			\sinh{(jz)} \sin{(j(x - t))},$$
		where $\epsilon = k'A'.$

		Parameters
		----------
		x, z : float or ndarray
			The horizontal and vertical position(s).
		t : float or ndarray, default=0
			The time(s) at which to evaluate the velocity.

		Returns
		-------
		ndarray
			1D array of `float` data, the vector components $u$ and $w$.
		"""
		h = self.wavenum * self.depth	# dimensionless water depth
		u = 1 - self.mean_speed
		w = np.zeros(len(z)) if isinstance(z, np.ndarray) else 0
		for i in range(1, self.order + 1):
			u_sum, w_sum = 0, 0
			for j in range(1, i + 1):
				u_sum += j * self.a[i, j] * np.cosh(j * (z + h)) \
						   * np.cos(j * (x - t))
				w_sum += j * self.a[i, j] * np.sinh(j * (z + h)) \
						   * np.sin(j * (x - t))
			u_sum *= self.steepness ** i
			w_sum *= self.steepness ** i
			u += u_sum
			w += w_sum
		return np.array([u, w])

	def partial_t(self, x, z, t): 
		r"""
		Compute the partial derivative of the fluid velocity with respect to
		time $\frac{\partial \boldsymbol{u}}{\partial t} = \Big\langle
		\frac{\partial u}{\partial t},\frac{\partial w}{\partial t}\Big\rangle$,
		
		$$\frac{\partial u}{\partial t} = \sum_{i=1}^{5} \epsilon^i
			\sum_{j=1}^i j^2 A_{ij} \cosh(jz) \sin(j(x - t)),$$
		$$ \frac{\partial w}{\partial t} = -\sum_{i=1}^{5} \epsilon^i
			\sum_{j=1}^i j^2 A_{ij} \sinh(jz) \cos(j(x - t)).$$

		Parameters
		----------
		x, z : float or ndarray
			The horizontal and vertical position(s).
		t : float or ndarray, default=0
			The time(s) at which to evaluate the derivative.

		Returns
		-------
		ndarray
			1D array of `float` data, the vector components of the derivative.
		"""
		h = self.wavenum * self.depth			# dimensionless water depth
		dtu = np.zeros(len(x)) if isinstance(x, np.ndarray) else 0
		dtw = np.zeros(len(z)) if isinstance(z, np.ndarray) else 0
		for i in range(1, self.order + 1):
			dtu_sum, dtw_sum = 0, 0
			for j in range(1, i + 1):
				dtu_sum += j * j * self.a[i, j] * np.cosh(j * (z + h)) \
							 * np.sin(j * (x - t))
				dtw_sum += j * j * self.a[i, j] * np.sinh(j * (z + h)) \
							 * np.cos(j * (x - t))
			dtu_sum *= self.steepness ** i
			dtw_sum *= self.steepness ** i
			dtu += dtu_sum
			dtw += dtw_sum
		return np.array([dtu, -dtw])

	def partial_x(self, x, z, t): 
		r"""
		Compute the partial derivative of the fluid with respect to the
		horizontal position
		$\frac{\partial \boldsymbol{u}}{\partial x} = \Big\langle
		\frac{\partial u}{\partial x},\frac{\partial w}{\partial x}\Big\rangle$,

		$$\frac{\partial u}{\partial x} = -\sum_{i=1}^{5} \epsilon^i
			\sum_{j=1}^i j^2 A_{ij} \cosh(jz) \sin(j(x - t)),$$
		$$\frac{\partial w}{\partial x} = \sum_{i=1}^{5} \epsilon^i
			\sum_{j=1}^i j^2 A_{ij} \sinh(jz) \cos(j(x - t)).$$

		Parameters
		----------
		x, z : float or ndarray
			The horizontal and vertical position(s).
		t : float or ndarray, default=0
			The time(s) at which to evaluate the derivative.

		Returns
		-------
		ndarray
			1D array of `float` data, the vector components of the derivative.
		"""
		h = self.wavenum * self.depth	# dimensionless water depth
		dxu = np.zeros(len(x)) if isinstance(x, np.ndarray) else 0
		dxw = np.zeros(len(z)) if isinstance(z, np.ndarray) else 0
		for i in range(1, self.order + 1):
			dxu_sum, dxw_sum = 0, 0
			for j in range(1, i + 1):
				dxu_sum += j * j * self.a[i, j] * np.cosh(j * (z + h)) \
							 * np.sin(j * (x - t))
				dxw_sum += j * j * self.a[i, j] * np.sinh(j * (z + h)) \
							 * np.cos(j * (x - t))
			dxu_sum *= self.steepness ** i
			dxw_sum *= self.steepness ** i
			dxu += dxu_sum
			dxw += dxw_sum
		return np.array([-dxu, dxw])

	def partial_z(self, x, z, t):
		r"""
		Compute the partial derivative of the fluid with respect to the
		vertical position, $\frac{\partial \boldsymbol{u}}{\partial z} = \Big\langle
		\frac{\partial u}{\partial z},\frac{\partial w}{\partial x}\Big\rangle$,

		$$\frac{\partial u}{\partial z} = \sum_{i=1}^{5} \epsilon^i
			\sum_{j=1}^i j^2 A_{ij} \sinh(jz) \cos(j(x - t)),$$ 
		$$\frac{\partial w}{\partial z} = \sum_{i=1}^{5} \epsilon^i
			\sum_{j=1}^i j^2 A_{ij} \cosh(jz) \sin(j(x - t)).$$

		Parameters
		----------
		x, z : float or ndarray
			The horizontal and vertical position(s).
		t : float or ndarray, default=0
			The time(s) at which to evaluate the derivative.

		Returns
		-------
		ndarray
			1D array of `float` data, the vector components of the derivative.
		"""
		h = self.wavenum * self.depth	# dimensionless water depth
		dzu = np.zeros(len(x)) if isinstance(x, np.ndarray) else 0
		dzw = np.zeros(len(z)) if isinstance(z, np.ndarray) else 0
		for i in range(1, self.order + 1):
			dzu_sum, dzw_sum = 0, 0
			for j in range(1, i + 1):
				dzu_sum += j * j * self.a[i, j] * np.sinh(j * (z + h)) \
							 * np.cos(j * (x - t))
				dzw_sum += j * j * self.a[i, j] * np.cosh(j * (z + h)) \
							 * np.sin(j * (x - t))
			dzu_sum *= self.steepness ** i
			dzw_sum *= self.steepness ** i
			dzu += dzu_sum
			dzw += dzw_sum
		return np.array([dzu, dzw])

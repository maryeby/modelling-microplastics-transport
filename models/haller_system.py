import numpy as np
import scipy.integrate as integrate

from transport_framework import transport_system

class HallerTransportSystem(transport_system.TransportSystem):
	"""Represent the transport of a particle in a linear water wave.[^1]"""

	def __init__(self, particle, flow, density_ratio):
		r"""
		Attributes
		----------
		particle : Particle (obj)
			The particle being transported.
		flow : Flow (obj)
			The flow through which the particle is transported.
		density_ratio : float
			The ratio $R$ between the particle and fluid densities,
			$$R = \frac{2\rho'_f}{\rho'_f + 2\rho'_p}.$$
		stokes_num : float
			The density-dependent Stokes number,
			$$St = \widehat{St} \Bigg(\frac 1R - \frac 12\Bigg).$$
		reynolds_num : float
			The particle Reynolds number, computed as,
			$$Re_p = \frac{2a'\omega'A'}{\nu'},$$
			where $\omega'$, $A'$ and $\nu'$ are attributes of the wave, and
			$a'$ is the radius of the particle.
		epsilon : float
			A relationship between the Stokes number $\widehat{St}$ and density
			ratio $R$, $$\epsilon = \frac{\widehat{St}}{R}.$$

		References
		----------
		[^1]: [G. Haller & T. Sapsis (2008).](
			  https://doi.org/10.1016/j.physd.2007.09.027)
			  Where do inertial particles go in fluid flows?
			  *Physica D: Nonlinear Phenomena* 237(5), 573–583.
		"""
		super().__init__(particle, flow, density_ratio)
		self.reynolds_num = (2 * self.flow.angular_freq * self.flow.amplitude
							   * np.sqrt(9 * self.particle.stokes_hat 
							   / (2 * self.flow.wavenum ** 2
							   * self.flow.reynolds_num))) \
							   / self.flow.kinematic_viscosity
		self.epsilon = self.particle.stokes_hat / self.density_ratio

	def set_stokes_num(self):
		r"""Set the density-dependent Stokes number $St$."""
		self.stokes_num = self.particle.stokes_hat * (1 / self.density_ratio
												   - 0.5)

	def maxey_riley(self, t, y):
		r"""
		Evaluate the Maxey-Riley equation without history effects.

		Parameters
		----------
		t : ndarray
			1D array containing `float` time series data.
		y : list
			A list of `float` data, the initial particle position and velocity.

		Returns
		-------
		ndarray
			1D array of `float` data, the particle velocity and acceleration.

		Notes
		-----
		Computations correspond to equation (3) in [1],
		$$\frac{\mathrm{d}\boldsymbol{x}}{\mathrm{d}t} = \boldsymbol{v},$$
		$$\frac{\mathrm{d}\boldsymbol{v}}{\mathrm{d}t} = \frac{\boldsymbol{u}
			- \boldsymbol{v}}{\epsilon}
			+ \frac{3R}{2} \frac{\mathrm{D}\boldsymbol{u}}{\mathrm{D}t}
			+ \Bigg(1 - \frac{3R}{2}\Bigg) \boldsymbol{g}$$ with
		$$R = \frac{2 \rho'_f}{\rho'_f + 2 \rho'_p},
			\qquad \epsilon = \frac{\widehat{St}}{R},
			\qquad \widehat{St} = \frac 29 (a'k')^2 Re,
			\qquad Re = \frac{\omega'}{k^{\prime 2} \nu'},$$
		where $Re$ is the Reynolds number, $a'$ is the particle radius, $\nu'$
		is the kinematic viscosity, $\omega'$ is the angular frequency, $k'$ is
		the wave number, and $\rho'$ is the density of the particle or the
		fluid, denoted by the subscript.
		"""
		R = self.density_ratio
		x, z = y[:2]
		particle_velocity = y[2:]

		stokes_drag = (self.flow.velocity(x, z, t) - particle_velocity) \
					  / self.epsilon
		buoyancy_force = (1 - 3 * R / 2) * self.flow.gravity
		fluid_pressure_gradient = 3 * R / 2 \
									* self.flow.material_derivative(x, z, t)
		particle_accel = stokes_drag + buoyancy_force + fluid_pressure_gradient

		return np.concatenate((particle_velocity, particle_accel))

	def inertial_equation(self, t, y, order):
		r"""
		Evaluate the inertial equation.[^1]

		Parameters
		----------
		t : ndarray
			1D array containing `float` time series data.
		y : list
			A list of `float` data, the initial particle position and velocity.
		order : int
			The order of the inertial equation (leading, first, or second).

		Returns
		-------
		x : ndarray
			1D array of `float` data, the horizontal particle position.
		z : ndarray
			1D array of `float` data, the vertical particle position.
		xdot : ndarray
			1D array of `float` data, the horizontal particle velocity.
		zdot : ndarray
			1D array of `float` data, the vertical particle velocity.
		t : ndarray
			1D array containing `float` time series data.

		Notes
		-----
		Computations correspond to equation (10) in [1],
		$$\boldsymbol{v} = \boldsymbol{u} + \epsilon \Bigg(\frac{3R}{2}]
		- 1\Bigg)\Bigg[\frac{\mathrm{D}\boldsymbol{u}}{\mathrm{D}t}
		- \boldsymbol{g}\Bigg] + \epsilon^2 \Bigg(1 - \frac{3R}{2}\Bigg)
		  \Bigg[\frac{\mathrm{D}^2\boldsymbol{u}}{\mathrm{D}t^2}
		+ \Bigg(\frac{\mathrm{D}\boldsymbol{u}}{\mathrm{D}t}
		- \boldsymbol{g}\Bigg) \cdot \nabla \boldsymbol{u}\Bigg]
		+ \mathcal{O}(\epsilon^3)$$ with
		$$R = \frac{2 \rho'_f}{\rho'_f + 2 \rho'_p},
			\qquad \epsilon = \frac{\widehat{St}}{R},
			\qquad \widehat{St} = \frac 29 (a'k')^2 Re,
			\qquad Re = \frac{\omega'}{k^{\prime 2} \nu'},$$
		where $Re$ is the Reynolds number, $a'$ is the particle radius, $\nu'$
		is the kinematic viscosity, $\omega'$ is the angular frequency, $k'$ is
		the wave number, and $\rho'$ is the density of the particle or the
		fluid, denoted by the subscript.
		"""
		g = self.flow.gravity
		R = self.density_ratio
		x, z = y[:2]
		fluid_velocity = self.flow.velocity(x, z, t)
		material_dv = self.flow.material_derivative(x, z, t)

		if order == 0:
			particle_velocity = fluid_velocity
		elif order == 1:
			particle_velocity = fluid_velocity + self.epsilon * (3 * R / 2 - 1)\
											   * (material_dv - g)
		elif order == 2:
			particle_velocity = fluid_velocity + self.epsilon * (3 * R / 2 - 1)\
								* (material_dv - g) \
								+ self.epsilon ** 2 * (1 - 3 * R / 2) \
								* (self.flow.material_derivative2(x, z, t)
								+ self.flow.dot_jacobian(material_dv - g,
														 x, z, t))
		else:
			print('Could not identify the order for the inertial equation.')

		stokes_drag = (fluid_velocity - particle_velocity) / self.epsilon
		buoyancy_force = (1 - 3 * R / 2) * g
		fluid_pressure_gradient = 3 * R / 2 * material_dv
		particle_accel = stokes_drag + buoyancy_force + fluid_pressure_gradient

		return np.concatenate((particle_velocity, particle_accel))

	def run_numerics(self, equation, x_0, z_0, num_periods, delta_t, order=2):
		"""
		Compute the position and velocity of the particle over time.

		Parameters
		----------
		equation : function
			The equation to evaluate, either M-R or the inertial equation.
		x_0, z_0 : float
			The initial horizontal and vertical position of the particle.
		num_periods : int
			The number of wave periods to integrate over.
		delta_t : float
			The size of the timesteps used for integration.
		order : int, default=2
			The order of the inertial equation (leading, first, or second).

		Returns
		-------
		x : ndarray
			1D array of `float` data, the horizontal particle position.
		z : ndarray
			1D array of `float` data, the vertical particle position.
		xdot : ndarray
			1D array of `float` data, the horizontal particle velocity.
		zdot : ndarray
			1D array of `float` data, the vertical particle velocity.
		t : ndarray
			1D array containing `float` time series data.

		Notes
		-----
		The initial velocity of the particle is set to the initial velocity of
		the fluid.
		"""
		# initialize local parameters
		t_final = num_periods * self.flow.period + delta_t
		t_span = (0, t_final)
		t_eval = np.arange(0, t_final, delta_t)
		xdot_0, zdot_0 = self.flow.velocity(x_0, z_0, 0)

		# run computations
		if 'maxey_riley' in str(equation):
			sols = integrate.solve_ivp(equation, t_span,
									   [x_0, z_0, xdot_0, zdot_0],
									   method='BDF', t_eval=t_eval,
									   rtol=1e-10, atol=1e-12)
		elif 'inertial' in str(equation):
			sols = integrate.solve_ivp(equation, t_span,
									   [x_0, z_0, xdot_0, zdot_0],
									   method='BDF', t_eval=t_eval,
									   rtol=1e-8, atol=1e-10, args=(order,))
		else:
			print('Could not recognize equation.')

		# unpack and return solutions
		x, z, xdot, zdot = sols.y
		t = sols.t
		return x, z, xdot, zdot, t

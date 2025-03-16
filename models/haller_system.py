import numpy as np
import scipy.integrate as integrate

from models import deep_water_wave
from transport_framework import particle, transport_system

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
			The ratio *R* between the particle and fluid densities.
		reynolds_num : float
			The particle Reynolds number, computed as,
			$$Re_p = \frac{2\omega'A'a'}{\nu'},$$
			where *ω'*, *A'* and ν' are attributes of the wave, and *a'* is the
			radius of the particle.
		epsilon : float
			A relationship between the Stokes number *St* and density ratio *R*,
			$$\epsilon = \frac{St}{R}.$$

		References
		----------
		[^1]: [G. Haller & T. Sapsis (2008).](
			  https://doi.org/10.1016/j.physd.2007.09.027)
			  Where do inertial particles go in fluid flows?
			  *Physica D: Nonlinear Phenomena* 237(5), 573–583.
		"""
		super().__init__(particle, flow, density_ratio)
		self.reynolds_num = (2 * self.flow.angular_freq * self.flow.amplitude
							   * np.sqrt(9 * self.particle.stokes_num 
							   / (2 * self.flow.wavenum ** 2
							   * self.flow.reynolds_num))) \
							   / self.flow.kinematic_viscosity
		self.epsilon = self.particle.stokes_num / self.density_ratio

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
		$$\frac{\mathrm{d}\mathbf{x}}{\mathrm{d}t} = \mathbf{v},$$
		$$\frac{\mathrm{d}\mathbf{v}}{\mathrm{d}t} = \frac{\mathbf{u}
			- \mathbf{v}}{\epsilon}
			+ \frac{3R}{2} \frac{\mathrm{D}\mathbf{u}}{\mathrm{D}t}
			+ \Bigg(1 - \frac{3R}{2}\Bigg) \mathbf{g}$$ with
		$$R = \frac{2 \rho'_f}{\rho'_f + 2 \rho'_p},
			\qquad \epsilon = \frac{St}{R},
			\qquad St = \frac 29 \Bigg(\frac{a'}{L'}\Bigg)^2 Re,
			\qquad Re = \frac{U'L'}{\nu'},$$
		where *a'* is the particle radius, *ν'* is the kinematic viscosity, *U'*
		and *L'* are the characteristic velocity and length scales respectively,
		and *ρ'* is the density of the particle or the fluid.

		References
		----------
		[^1]: [G. Haller & T. Sapsis (2008).](
			  https://doi.org/10.1016/j.physd.2007.09.027)
			  Where do inertial particles go in fluid flows?
			  *Physica D: Nonlinear Phenomena* 237(5), 573–583.
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
		$$\mathbf{v} = \mathbf{u} + \epsilon \Bigg(\frac{3R}{2} - 1\Bigg)
		\Bigg[\frac{\mathrm{D}\mathbf{u}}{\mathrm{D}t} - \mathbf{g}\Bigg]
		+ \epsilon^2 \Bigg(1 - \frac{3R}{2}\Bigg)
		\Bigg[\frac{\mathrm{D}^2\mathbf{u}}{\mathrm{D}t^2}
		+ \Bigg(\frac{\mathrm{D}\mathbf{u}}{\mathrm{D}t} - \mathbf{g}\Bigg)
		\cdot \nabla \mathbf{u}\Bigg]
		+ \mathcal{O}(\epsilon^3)$$ with
		$$R = \frac{2 \rho'_f}{\rho'_f + 2 \rho'_p},
			\qquad \epsilon = \frac{St}{R},
			\qquad St = \frac 29 \Bigg(\frac{a'}{L'}\Bigg)^2 Re,
			\qquad Re = \frac{U'L'}{\nu'},$$
		where *a* is the particle radius, *ν'* is the kinematic viscosity, *U'*
		and *L'* are the characteristic velocity and length scales respectively,
		and *ρ'* is the density of the particle or the fluid.

		References
		----------
		[^1]: [G. Haller & T. Sapsis (2008).](
			  https://doi.org/10.1016/j.physd.2007.09.027)
			  Where do inertial particles go in fluid flows?
			  *Physica D: Nonlinear Phenomena* 237(5), 573–583.
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

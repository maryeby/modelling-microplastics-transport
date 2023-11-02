import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
import scipy.integrate as integrate
from models import deep_water_wave
from transport_framework import particle, transport_system

class HallerTransportSystem(transport_system.TransportSystem):
	""" 
	Represents the transport of an inertial particle in a linear water wave as
	described by Haller and Sapsis (2008).
	"""

	def __init__(self, particle, flow, density_ratio):
		r"""
		Attributes
		----------
		particle : Particle (obj)
			The particle being transported.
		flow : Flow (obj)
			The flow through which the particle is transported.
		density_ratio : float
			The ratio between the particle and fluid densities.
		epsilon : float
			A relationship between the Stokes number *St* and density ratio *R*,
			$$\epsilon = \frac{St}{R}.$$
		"""
		super().__init__(particle, flow, density_ratio)
		self.epsilon = self.particle.stokes_num / self.density_ratio

	def maxey_riley(self, t, y):
		r"""
		Evaluates the Maxey-Riley equation without history effects,
		corresponding to equation (3) in Haller and Sapsis (2008),
		$$\frac{\mathrm{d}\mathbf{x}}{\mathrm{d}t} = \mathbf{v},$$
		$$\frac{\mathrm{d}\mathbf{v}}{\mathrm{d}t} = \frac{\mathbf{u}
			- \mathbf{v}}{\epsilon}
			+ \frac{3R}{2} \frac{\mathrm{D}\mathbf{u}}{\mathrm{D}t}
			+ \Bigg(1 - \frac{3R}{2}\Bigg) \mathbf{g}$$ with
		$$R = \frac{2 \rho_f}{\rho_f + 2 \rho_p},
			\qquad \epsilon = \frac{St}{R},
			\qquad St = \frac 29 \Bigg(\frac{a}{L}\Bigg)^2 Re,
			\qquad Re = \frac{UL}{\nu},$$
		where *a* is the particle radius, *ν* is the kinematic viscosity, *U*
		and *L* are the characteristic velocity and length scales respectively,
		and *ρ* is the density of the particle or the fluid.
		
		Parameters
		----------
		t : float or array
			The time(s) when the Maxey-Riley equation should be evaluated.
		y : list (array-like)
			A list containing the initial particle position and velocity.

		Returns
		-------
		Array
			The components of the particle's velocity and acceleration.
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
		Evalutes the inertial equation, corresponding to equation (10) in
		Haller and Sapsis (2008),
		$$\mathbf{v} = \mathbf{u} + \epsilon \Bigg(\frac{3R}{2} - 1\Bigg)
		\Bigg[\frac{\mathrm{D}\mathbf{u}}{\mathrm{D}t} - \mathbf{g}\Bigg]
		+ \epsilon^2 \Bigg(1 - \frac{3R}{2}\Bigg)
		\Bigg[\frac{\mathrm{D}^2\mathbf{u}}{\mathrm{D}t^2}
		+ \Bigg(\frac{\mathrm{D}\mathbf{u}}{\mathrm{D}t} - \mathbf{g}\Bigg)
		\cdot \nabla \mathbf{u}\Bigg]
		+ \mathcal{O}(\epsilon^3)$$ with
		$$R = \frac{2 \rho_f}{\rho_f + 2 \rho_p},
			\qquad \epsilon = \frac{St}{R},
			\qquad St = \frac 29 \Bigg(\frac{a}{L}\Bigg)^2 Re,
			\qquad Re = \frac{UL}{\nu},$$
		where *a* is the particle radius, *ν* is the kinematic viscosity, *U*
		and *L* are the characteristic velocity and length scales respectively,
		and *ρ* is the density of the particle or the fluid.

		Parameters
		----------
		t : float
			The time(s) when the inertial equation should be evaluated.
		y : list (array-like)
			A list containing the x and z values to use in the computations.
		order : int
			The order of the inertial equation (leading, first, or second).

		Returns
		-------
		Array
			The components of the particle's velocity and acceleration.
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

	def run_numerics(self, equation, x_0, z_0, num_periods, delta_t, order=2,
					 method='BDF'):
		"""
		Computes the position and velocity of the particle over time.

		Parameters
		----------
		equation : fun
			The equation to evaluate, either M-R or the inertial equation.
		x_0 : float
			The initial horizontal position of the particle.
		z_0 : float
			The initial vertical position of the particle.
		num_periods : int
			The number of wave periods to integrate over.
		delta_t : float
			The size of the timesteps used for integration.
		order : int, default=2
			The order of the inertial equation (leading, first, or second).
		method : str, default='BDF'
			The method of integration to use.

		Returns
		-------
		x : array
			The horizontal positions of the particle.
		z : array
			The vertical positions of the particle.
		xdot : array
			The horizontal velocities of the particle.
		zdot : array
			The vertical velocities of the particle.
		t : array
			The times at which the model was evaluated.

		Notes
		-----
		The velocity of the particle is set to the initial velocity of the
		fluid.
		"""
		# initialize local parameters
		t_final = num_periods * self.flow.period
		t_span = (0, t_final)
		t_eval = np.arange(0, t_final, delta_t)
		xdot_0, zdot_0 = self.flow.velocity(x_0, z_0, 0)

		# run computations
		if 'maxey_riley' in str(equation):
			sols = integrate.solve_ivp(equation, t_span,
									   [x_0, z_0, xdot_0, zdot_0],
									   method=method, t_eval=t_eval,
									   rtol=1e-10, atol=1e-12)
		elif 'inertial' in str(equation):
			sols = integrate.solve_ivp(equation, t_span,
									   [x_0, z_0, xdot_0, zdot_0],
									   method=method, t_eval=t_eval,
									   rtol=1e-8, atol=1e-10, args=(order,))
		else:
			print('Could not recognize equation.')

		# unpack and return solutions
		x, z, xdot, zdot = sols.y
		t = sols.t
		return x, z, xdot, zdot, t

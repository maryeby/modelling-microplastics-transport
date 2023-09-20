import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
import scipy.integrate as integrate
from models import dim_deep_water_wave
from transport_framework import particle, transport_system

class SantamariaTransportSystem(transport_system.TransportSystem):
	""" 
	Represents the transport of an inertial particle in a linear water wave as
	described by Santamaria et al. (2013).
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
		st_response_time : float
			The Stokes response time τ, computed as
			$$\tau = \frac{St}{\omega}.$$
		"""
		super().__init__(particle, flow, density_ratio)
		self.st_response_time = self.particle.stokes_num \
								/ self.flow.angular_freq

	def maxey_riley(self, t, y):
		r"""
		Evaluates the Maxey-Riley equation without history effects,
		corresponding to equations (3) and (4) in Santamaria et al. (2013),
		$$\frac{\mathrm{d}\mathbf{x}}{\mathrm{d}t} = \mathbf{v},$$
		$$\frac{\mathrm{d}\mathbf{v}}{\mathrm{d}t}
			= \frac{\mathbf{u} - \mathbf{v}}{\tau} + (1 - \beta) \mathbf{g}
			+ \beta \frac{\mathrm{D}\mathbf{u}}{\mathrm{D}t}$$ with
		$$\tau = \frac{a^2}{3 \beta \nu},
			\qquad \beta = \frac{3 \rho_f}{\rho_f + 2 \rho_p},$$
		where *a* is the particle radius, *ν* is the kinematic viscosity, and
		*ρ* is the density of the particle or the fluid.
		
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
		beta = self.density_ratio
		tau = self.st_response_time
		x, z = y[:2]
		particle_velocity = y[2:]

		stokes_drag = (self.flow.velocity(x, z, t) - particle_velocity) / tau
		buoyancy_force = (1 - beta) * self.flow.gravity
		fluid_pressure_gradient = beta * self.flow.material_derivative(x, z, t)
		particle_accel = stokes_drag + buoyancy_force + fluid_pressure_gradient

		return np.concatenate((particle_velocity, particle_accel))

	def inertial_equation(self, t, y, order):
		r"""
		Evalutes the inertial equation, corresponding to equation (5) in
		Santamaria et al. (2013),
		$$\mathbf{v} = \mathbf{u} + \tau (1 - \beta) \Bigg(\mathbf{g}
		- \frac{\mathrm{D}\mathbf{u}}{\mathrm{D}t}\Bigg)
		+ \tau^2 (1 - \beta) \frac{\mathrm{D}^2\mathbf{u}}{\mathrm{D}t^2}
		+ \mathcal{O}(\tau^3)$$ with
		$$\tau = \frac{a^2}{3 \beta \nu},
			\qquad \beta = \frac{3 \rho_f}{\rho_f + 2 \rho_p},$$
		where *a* is the particle radius, *ν* is the kinematic viscosity, and
		*ρ* is the density of the particle or the fluid.

		Parameters
		----------
		t : float or array
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
		beta = self.density_ratio
		tau = self.st_response_time
		g = self.flow.gravity
		x, z = y[:2]
		fluid_velocity = self.flow.velocity(x, z, t)
		material_dv = self.flow.material_derivative(x, z, t)

		if order == 0:
			particle_velocity = fluid_velocity
		elif order == 1:
			particle_velocity = fluid_velocity + tau * (1 - beta) \
											   * (g - material_dv)
		elif order == 2:
			particle_velocity = fluid_velocity + tau * (1 - beta) \
								* (g - material_dv) \
								+ tau ** 2 * (1 - beta) \
								* self.flow.material_derivative2(x, z, t)
		else:
			print('Could not identify the order for the inertial equation.')

		stokes_drag = (fluid_velocity - particle_velocity) / tau 
		buoyancy_force = (1 - beta) * g 
		fluid_pressure_gradient = beta * material_dv
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
		# initial parameters
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

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
		"""
		Attributes
		----------
		particle : Particle (obj)
			The particle being transported.
		flow : Flow (obj)
			The flow through which the particle is transported.
		density_ratio : float
			The ratio between the particle's density and the fluid's density.
		epsilon : float
			A relationship between the Stokes number *St* and density ratio *R*.
		"""
		super().__init__(particle, flow, density_ratio)
		self.epsilon = self.particle.stokes_num / self.density_ratio

	def maxey_riley(self, t, y):
		r"""
		Evaluates the Maxey-Riley equation without the history term,
		corresponding to equation (3) in Haller and Sapsis (2008),
		$$\frac{\mathrm{d}\textbf{x}}{\mathrm{d}t} = \textbf{v},$$
		$$\frac{\mathrm{d}\textbf{v}}{\mathrm{d}t} = \frac{\textbf{u}
			- \textbf{v}}{\epsilon}
			+ \frac{3R}{2} \frac{\mathrm{d}\textbf{u}}{\mathrm{d}t}
			+ (1 - \frac{3R}{2}) \textbf{g}.$$
		
		Parameters
		----------
		t : float
			The time to use in the computations
		y : list (array-like)
			A list containing the x, z, xdot, zdot values to use.

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
		$$\textbf{v} = \textbf{u} + \epsilon \Bigg(\frac{3R}{2} - 1\Bigg)
		\Bigg[\frac{\mathrm{D}\textbf{u}}{\mathrm{D}t} - \textbf{g}\Bigg]
		+ \epsilon^2 \Bigg(1 - \frac{3R}{2}\Bigg)
		\Bigg[\frac{\mathrm{D}^2\textbf{u}}{\mathrm{D}t^2}
		+ \Bigg(\frac{\mathrm{D}\textbf{u}}{\mathrm{D}t} - \textbf{g}\Bigg)
		\cdot \nabla \textbf{u}\Bigg]
		+ \mathcal{O}(\epsilon^3).$$

		Parameters
		----------
		t : float
			The time to use in the computations
		y : list (array-like)
			A list containing the x and z values to use, and the order of the
			equation.
		order : int
			The order at which to evaluate the inertial equation.

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
			u, w = fluid_velocity
			Du, Dw = material_dv
			gx, gz = g
			jacobian_term = np.array([Dw * u - Du * w, Du * u + Dw * w]) \
							- np.array([u * gz - w * gx, u * gx + w * gz])
			particle_velocity = fluid_velocity + self.epsilon * (3 * R / 2 - 1)\
								* (material_dv - g) \
								+ self.epsilon ** 2 * (1 - 3 * R / 2) \
								* (self.flow.material_derivative2(x, z, t)
								+ jacobian_term)
		else:
			print('Could not identify the order for the inertial equation.')

		stokes_drag = (fluid_velocity - particle_velocity) / self.epsilon
		buoyancy_force = (1 - 3 * R / 2) * g
		fluid_pressure_gradient = 3 * R / 2 * material_dv
		particle_accel = stokes_drag + buoyancy_force + fluid_pressure_gradient

		return np.concatenate((particle_velocity, particle_accel))

	def run_numerics(self, equation, order=2, x_0=0, z_0=0, num_periods=50,
					 delta_t=1e-3, method='BDF'):
		"""
		Computes the position and velocity of the particle over time.

		Parameters
		----------
		equation : fun
			The equation to evaluate, either M-R or the inertial equation.
		order : int
			The order at which to evaluate the inertial equation.
		x_0 : float, default=0
			The initial horizontal position of the particle.
		z_0 : float, default=0
			The initial vertical position of the particle.
		num_periods : int, default=50
			The number of oscillation periods to integrate over.
		delta_t : float, default=5e-3
			The size of the timesteps of integration.
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
		num_steps = int(np.rint(num_periods * self.flow.period / delta_t))
		t_span = (0, num_periods * self.flow.period)
		t_eval = np.linspace(0, num_periods * self.flow.period, num_steps)
		xdot_0, zdot_0 = self.flow.velocity(x_0, z_0, 0)

		# run computations
		if 'maxey_riley' in str(equation):
			sols = integrate.solve_ivp(equation, t_span,
									   [x_0, z_0, xdot_0, zdot_0],
									   method=method, t_eval=t_eval,
									   rtol=1e-8, atol=1e-10)
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

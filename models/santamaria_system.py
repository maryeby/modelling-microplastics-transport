import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
from models import dim_deep_water_wave
from transport_framework import particle, transport_system

class SantamariaTransportSystem(transport_system.TransportSystem):
	""" 
	Represents the transport of an inertial particle in a linear water wave as
	described by Santamaria et al. (2013).
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
		st_response_time : float
			The Stokes response time tau.
		"""
		super().__init__(particle, flow, density_ratio)
		self.st_response_time = self.particle.stokes_num \
								/ self.flow.angular_freq

	def maxey_riley(self, t, y):
		r"""
		Evaluates the Maxey-Riley equation without the history term,
		corresponding to equations (3) and (4) in Santamaria et al. (2013),
		$$\frac{\mathrm{d}\textbf{x}}{\mathrm{d}t} = \textbf{V},$$
		$$\frac{\mathrm{d}\textbf{V}}{\mathrm{d}t}
			= \frac{\textbf{u} - \textbf{V}}{\tau} + (1 - \beta) \textbf{g}
			+ \beta \frac{\mathrm{d}\textbf{u}}{\mathrm{d}t}.$$
		
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
		$$\textbf{V} = \textbf{u} + \tau (1 - \beta) \Bigg(\textbf{g}
		- \frac{\mathrm{d}\textbf{u}}{\mathrm{d}t}\Bigg)
		+ \tau^2 (1 - \beta) \frac{\mathrm{d}^2\textbf{u}}{\mathrm{d}t^2}
		+ \mathcal{O}(\tau^3).$$

		Parameters
		----------
		t : float
			The time to use in the computations
		y : list (array-like)
			A list containing the x and z values to use, and the order of the
			equation.
		order : int
			The order at which to evalaute the inertial equation.

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
		particle_velocity = np.zeros_like(fluid_velocity)
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

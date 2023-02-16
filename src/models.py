import numpy as np
from scipy import constants

def mr_no_history(t, y, transport_sys): 
	"""
	Evaluates the M-R equation without the history term.

	Parameters
	----------
	transport_sys : TransportSystem (obj)
		The inertial particle transport system to model.
	t : float
		The time to use in the computations.
	y : list (array-like)
		A list containing the x, z, xdot, zdot values to use in the evaluation.
	
	Returns
	-------
	Array
		The components of the particle's velocity and acceleration.
	"""
	R = transport_sys.get_density()
	St = transport_sys.get_stokes_num()
	x, z = y[:2]                # current position
	particle_velocity = y[2:]   # components of particle velocity, xdot and zdot

	# compute the terms of the Maxey-Riley equation (neglecting history)
	fluid_pressure_gradient = 3 / 2 * R \
								* transport_sys.material_derivative(x, z, t)
	# TODO: check if the vector is correct for g
	buoyancy_force = (1 - 3 / 2 * R) * np.array([0, -constants.g])
	stokes_drag = R / St * (particle_velocity 
							- transport_sys.fluid_velocity(x, z, t))

	# particle acceleration is the LHS of the M-R equation, denoted dV/dt
	particle_acceleration = fluid_pressure_gradient + buoyancy_force \
													- stokes_drag

	return np.concatenate((particle_velocity, particle_acceleration))

def mr_with_history(t, y, transport_sys):
	"""
	Evaluates the M-R equation with the history term.

	Parameters
	----------
	transport_sys : TransportSystem (obj)
		The inertial particle transport system to model.
	t : float
		The time to use in the computations.
	y : list (array-like)
		A list containing the x, z, xdot, zdot values to use in the evaluation.
	
	Returns
	-------
	Array
		The components of the particle's velocity and acceleration.
	"""
	R = transport_sys.get_density()
	St = transport_sys.get_stokes_num()
	x, z = y[:2]                # current position
	particle_velocity = y[2:]   # u and w components of particle velocity

	# compute the terms of the Maxey-Riley equation except history
	fluid_pressure_gradient = 3 / 2 * R \
								* transport_sys.material_derivative(x, z, t)
	# TODO: check if the vector is correct for g
	buoyancy_force = (1 - 3 / 2 * R) * np.array([0, -constants.g])
	stokes_drag = R / St * (particle_velocity
							- transport_sys.fluid_velocity(x, z, t))

	# compute the history term
	integrand = np.empty(2,)
	coefficient = np.sqrt(9 / (2 * np.pi)) * (R / np.sqrt(St))
	kernel = np.array(transport_sys.get_particle_history()) \
			 - np.array(transport_sys.get_fluid_history())
	num_steps = len(transport_sys.get_timesteps()) - 1
	alpha = 0
	delta_t = 0

	for j in range(num_steps):
		if j == 0:
			alpha = 4 / 3
		elif j == num_steps:
			alpha = (4 / 3) * (num_steps - 1) ** (3 / 2) \
							- num_steps ** (3 / 2) + (3 / 2) \
							* np.sqrt(num_steps)
			delta_t = np.abs(self.__timesteps[j] - self.__timesteps[j - 1])
		else:
			alpha = (4 / 3) * (j - 1) ** (3 / 2) + (j + 1) ** (3 / 2) - 2 \
							* j ** (3 / 2)
			delta_t = np.abs(self.__timesteps[j] - self.__timesteps[j - 1])
		integrand += np.sqrt(delta_t) * alpha * kernel[num_steps - j]
	history = coefficient * integrand

	# particle acceleration is the LHS of the M-R equation, denoted dV/dt
	particle_acceleration = fluid_pressure_gradient + buoyancy_force \
													- stokes_drag - history
	# update relevant TransportSystem attributes to keep track of history
	transport_sys.update_particle_history(particle_acceleration)
	#TODO: add fluid_accel function to TransportSystem class
	transport_sys.update_fluid_history(transport_sys.fluid_accel(x, z, t))
	transport_sys.update_timesteps(t)

	return np.concatenate((particle_velocity, particle_acceleration))

def santamaria(t, y, transport_sys):
	"""
	Evaluates equation (4) from Santamaria et al., 2013.

	Parameters
	----------
	transport_sys : TransportSystem (obj)
		The inertial particle transport system to model.
	t : float
		The time to use in the computations.
	y : list (array-like)
		A list containing the x, z, xdot, zdot values to use in the evaluation.
	
	Returns
	-------
	Array
		The components of the particle's velocity and acceleration.
	"""
	beta = transport_sys.get_beta()
	tau = transport_sys.get_response_time()
	x, z = y[:2]
	particle_velocity = y[2:]

	stokes_drag = (transport_sys.fluid_velocity(x, z, t, deep=True) \
				   - particle_velocity) / tau
	buoyancy_force = (1 - beta) * np.array([0, -constants.g])
	fluid_pressure_gradient = beta * transport_sys.material_derivative(x, z, t,
																  deep=True)
	particle_accel = stokes_drag + buoyancy_force + fluid_pressure_gradient

	return np.concatenate((particle_velocity, particle_accel))

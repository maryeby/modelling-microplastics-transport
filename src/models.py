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
	particle_velocity = y[2:]   # particle velocity V

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
	Evaluates equations (3) and (4) from Santamaria et al., 2013.

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

	stokes_drag = (transport_sys.fluid_velocity(x, z, t, deep=True,
												dimensional=True) \
				   - particle_velocity) / tau
	buoyancy_force = (1 - beta) * np.array([0, -constants.g])
	fluid_pressure_gradient = beta * transport_sys.material_derivative(x, z, t,
												   deep=True, dimensional=True)
	particle_accel = stokes_drag + buoyancy_force + fluid_pressure_gradient

	return np.concatenate((particle_velocity, particle_accel))

def santamaria_order0(t, y, transport_sys):
	"""
	Evaluates equations (3)-(5) from Santamaria et al., 2013 to leading order.

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
	fluid_velocity = transport_sys.fluid_velocity(x, z, t, deep=True,
												  dimensional=True)
	particle_velocity = fluid_velocity

	stokes_drag = (fluid_velocity - particle_velocity) / tau
	buoyancy_force = (1 - beta) * np.array([0, -constants.g])
	fluid_pressure_gradient = beta * transport_sys.material_derivative(x, z, t,
												   deep=True, dimensional=True)
	particle_accel = stokes_drag + buoyancy_force + fluid_pressure_gradient

	return np.concatenate((particle_velocity, particle_accel))

def santamaria_order1(t, y, transport_sys):
	"""
	Evaluates equations (3)-(5) from Santamaria et al., 2013 to first order.

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
	U = transport_sys.get_max_velocity()
	k = transport_sys.get_wavenum()
	g = np.array([0, -constants.g])

	x, z = y[:2]
	material_derivative = transport_sys.material_derivative(x, z, t, deep=True,
														dimensional=True)
	fluid_velocity = transport_sys.fluid_velocity(x, z, t, deep=True,
												  dimensional=True)
	particle_velocity = fluid_velocity + tau * (1 - beta) \
									   * (g - material_derivative)

	stokes_drag = (fluid_velocity - particle_velocity) / tau
	buoyancy_force = (1 - beta) * g
	fluid_pressure_gradient = beta * material_derivative
	particle_accel = stokes_drag + buoyancy_force + fluid_pressure_gradient

	return np.concatenate((particle_velocity, particle_accel))

def santamaria_order2(t, y, transport_sys):
	"""
	Evaluates equations (3)-(5) from Santamaria et al., 2013 to second order.

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
	U = transport_sys.get_max_velocity()
	k = transport_sys.get_wavenum()
	g = np.array([0, -constants.g])

	x, z = y[:2]
	material_derivative = transport_sys.material_derivative(x, z, t, deep=True,
															dimensional=True)
	fluid_velocity = transport_sys.fluid_velocity(x, z, t, deep=True,
												  dimensional=True)
	Du, Dw = material_derivative
	u, w = fluid_velocity
	new_term = k * np.array([Du * -w + Dw * u, Du * u + Dw * w])
	particle_velocity = fluid_velocity + tau * (1 - beta) \
									   * (g - material_derivative) \
							  		   + tau ** 2 \
									   * (1 - beta) \
									   * (transport_sys.material_derivative2(
													    x, z, t, deep=True,
													    dimensional=True)
									   + new_term) \

	stokes_drag = (fluid_velocity - particle_velocity) / tau
	buoyancy_force = (1 - beta) * g
	fluid_pressure_gradient = beta * material_derivative
	particle_accel = stokes_drag + buoyancy_force + fluid_pressure_gradient

	return np.concatenate((particle_velocity, particle_accel))

def haller(t, y, transport_sys):
	"""
	Evaluates equation (3) from Haller 2007.

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
	epsilon = transport_sys.get_stokes_num() / R
	g = np.array([0, -constants.g])
	x, z = y[:2]
	particle_velocity = y[2:]

	stokes_drag = (transport_sys.fluid_velocity(x, z, t, deep=True)
				   - particle_velocity) / epsilon
	buoyancy_force = (1 - 3 * R / 2) * g
	fluid_pressure_gradient = 3 * R / 2 \
								* transport_sys.material_derivative(x, z, t,
																	deep=True)
	particle_accel = stokes_drag + buoyancy_force + fluid_pressure_gradient

	return np.concatenate((particle_velocity, particle_accel))

def haller_order0(t, y, transport_sys):
	"""
	Evaluates equations (3) and (12) from Haller 2007 to leading order.

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
	epsilon = transport_sys.get_stokes_num() / R
	g = np.array([0, -constants.g])

	x, z = y[:2]
	fluid_velocity = transport_sys.fluid_velocity(x, z, t, deep=True)
	particle_velocity = fluid_velocity

	stokes_drag = (fluid_velocity - particle_velocity) / epsilon
	buoyancy_force = (1 - 3 * R / 2) * g
	fluid_pressure_gradient = 3 * R / 2 \
								* transport_sys.material_derivative(x, z, t,
																	deep=True)
	particle_accel = stokes_drag + buoyancy_force + fluid_pressure_gradient

	return np.concatenate((particle_velocity, particle_accel))

def haller_order1(t, y, transport_sys):
	"""
	Evaluates equations (3) and (12) from Haller 2007 to first order.

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
	epsilon = transport_sys.get_stokes_num() / R
	g = np.array([0, -constants.g])

	x, z = y[:2]
	material_derivative = transport_sys.material_derivative(x, z, t, deep=True)
	fluid_velocity = transport_sys.fluid_velocity(x, z, t, deep=True)
	particle_velocity = fluid_velocity + epsilon * (3 * R / 2 - 1) \
									   * (material_derivative - g)

	stokes_drag = (fluid_velocity - particle_velocity) / epsilon
	buoyancy_force = (1 - 3 * R / 2) * g
	fluid_pressure_gradient = 3 * R / 2 * material_derivative
	particle_accel = stokes_drag + buoyancy_force + fluid_pressure_gradient

	return np.concatenate((particle_velocity, particle_accel))

def haller_order2(t, y, transport_sys):
	"""
	Evaluates equations (3) and (12) from Haller 2007 to second order.

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
	epsilon = transport_sys.get_stokes_num() / R
	g = np.array([0, -constants.g])

	x, z = y[:2]
	material_derivative = transport_sys.material_derivative(x, z, t, deep=True)
	fluid_velocity = transport_sys.fluid_velocity(x, z, t, deep=True)
	
	u, w = fluid_velocity
	Du, Dw = material_derivative
	gx, gz = g
	new_term = np.array([Du * -w + Dw * u, Du * u + Dw * w]) \
			   - np.array([u * gz - w * gx, u * gx + w * gz])

	particle_velocity = fluid_velocity + epsilon * (3 * R / 2 - 1) \
									   * (material_derivative - g) \
									   + epsilon ** 2 * (1 - 3 * R / 2) \
									   * (transport_sys.material_derivative2(
														x, z, t, deep=True)
									   + new_term)

	stokes_drag = (fluid_velocity - particle_velocity) / epsilon
	buoyancy_force = (1 - 3 * R / 2) * g
	fluid_pressure_gradient = 3 * R / 2 * material_derivative
	particle_accel = stokes_drag + buoyancy_force + fluid_pressure_gradient

	return np.concatenate((particle_velocity, particle_accel))

def haller_order2_no_jacobian(t, y, transport_sys):
	"""
	Evaluates equations (3) and (12) from Haller 2007 to second order.

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

	Notes
	-----
	This evaluation excludes the contribution of the Jacobian term from (12).
	"""
	R = transport_sys.get_density()
	epsilon = transport_sys.get_stokes_num() / R
	g = np.array([0, -constants.g])

	x, z = y[:2]
	material_derivative = transport_sys.material_derivative(x, z, t, deep=True)
	fluid_velocity = transport_sys.fluid_velocity(x, z, t, deep=True)

	particle_velocity = fluid_velocity + epsilon * (3 * R / 2 - 1) \
									   * (material_derivative - g) \
									   + epsilon ** 2 * (1 - 3 * R / 2) \
									   * transport_sys.material_derivative2(
														x, z, t, deep=True)

	stokes_drag = (fluid_velocity - particle_velocity) / epsilon
	buoyancy_force = (1 - 3 * R / 2) * g
	fluid_pressure_gradient = 3 * R / 2 * material_derivative
	particle_accel = stokes_drag + buoyancy_force + fluid_pressure_gradient

	return np.concatenate((particle_velocity, particle_accel))

def modified_santamaria(t, y, transport_sys):
	"""
	Numerically solves equations (6*) and (7*).

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
	bprime = 1 - beta
	St = transport_sys.get_stokes_num()
	tau = transport_sys.get_response_time()
	epsilon = transport_sys.get_wavenum() * transport_sys.get_amplitude()
	g = np.array([0, -constants.g])

	x, z = y[:2]
	material_derivative = transport_sys.material_derivative(x, z, t, deep=True)
	fluid_velocity = transport_sys.fluid_velocity(x, z, t, deep=True)

	u, w = fluid_velocity
	xdot = u - St * bprime * w - St ** 2 * bprime * u \
			 + epsilon * St ** 2 * bprime * (np.exp(2 * z) - w ** 2 + w * u)
	zdot = (-St * bprime) / epsilon + w + St * bprime * u \
				+ St ** 2 * bprime * u \
				+ epsilon * (-St * bprime * np.exp(2 * z) + St ** 2 * bprime
				* (2 * np.exp(2 * z) * w - u ** 2 - u * w))
	particle_velocity = np.array([xdot, zdot])

	stokes_drag = (fluid_velocity - particle_velocity) / tau
	buoyancy_force = (1 - beta) * g
	fluid_pressure_gradient = beta * material_derivative
	particle_accel = stokes_drag + buoyancy_force + fluid_pressure_gradient

	return np.concatenate((particle_velocity, particle_accel))

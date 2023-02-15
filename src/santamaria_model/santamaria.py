import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate 
from scipy.optimize import curve_fit

class SantamariaModel:
	def __init__(self, wave_num, amplitude, stokes_num, beta, gravity=9.8):
		self.__wave_num = wave_num		# k
		self.__amplitude = amplitude	# A
		self.__stokes_num = stokes_num	# St
		self.__beta = beta				# beta
		self.__gravity = gravity		# g
		self.__angular_freq = np.sqrt(self.__gravity * self.__wave_num) # omega
		self.__period = 2 * np.pi / self.__angular_freq
		self.__max_velocity = self.__angular_freq * self.__amplitude	  # U
		self.__st_response_time = self.__stokes_num / self.__angular_freq # tau
		self.__phase_velocity = self.__angular_freq / self.__wave_num
		self.__settling_velocity = -(1 - self.__beta) * self.__gravity \
					   			   * self.__st_response_time

	def get_wave_num(self):
		return self.__wave_num

	def get_angular_freq(self):
		return self.__angular_freq

	def get_max_velocity(self):
		return self.__max_velocity

	def get_settling_velocity(self):
		return self.__settling_velocity

	def fluid_velocity(self, x, z, t):
		return np.array([self.__max_velocity * np.exp(self.__wave_num * z) 
						 * np.cos(self.__wave_num * x
								  - self.__angular_freq * t),
						 self.__max_velocity * np.exp(self.__wave_num * z) 
						 * np.sin(self.__wave_num * x
								  - self.__angular_freq * t)])

	def fluid_derivative(self, x, z, t):
		u, w = self.fluid_velocity(x, z, t)
		return np.array([self.__angular_freq * w,
						 self.__wave_num * self.__max_velocity ** 2
										 * np.exp(2 * self.__wave_num * z)
										 - self.__angular_freq * u])
#		return np.array([self.__angular_freq * self.__max_velocity
#						* np.exp(self.__wave_num * z)
#						* np.sin(self.__wave_num * x - self.__angular_freq * t),
#						-self.__angular_freq * self.__max_velocity
#						* np.exp(self.__wave_num * z)
#						* np.cos(self.__wave_num * x - self.__angular_freq * t)]
#						+ np.dot(fluid_velocity,
#								 np.gradient(1/2 * fluid_velocity ** 2)))

	def analytical_particle_velocity(self, x_0=0, z_0=0, t=0):
		# local variables for equations (11) and (12)
		bprime = 1 - self.__beta
		phi = self.__wave_num * x_0 - self.__angular_freq * t
		z_0t = z_0 - self.__stokes_num * self.__phase_velocity * bprime * t

		# equation (11)
		u = self.__max_velocity * np.exp(self.__wave_num * z_0t) \
								* ((1 - self.__stokes_num ** 2 * self.__beta
									  * bprime) * np.cos(phi) \
									  - self.__stokes_num * bprime \
									  * np.sin(phi)) \
								+ self.__max_velocity ** 2 \
								/ self.__phase_velocity \
								* np.exp(2 * self.__wave_num * z_0t) \
								* (1 - self.__stokes_num ** 2 * self.__beta
									 * bprime)
		# equation (12)
		w = self.__max_velocity * np.exp(self.__wave_num * z_0t) \
								* ((1 - self.__stokes_num ** 2 * self.__beta
									  * bprime)
								* np.sin(phi) + self.__stokes_num * bprime \
								* np.cos(phi)) \
								- self.__phase_velocity \
								* self.__stokes_num * bprime \
								* (1 + 2 * self.__max_velocity ** 2
								/ self.__phase_velocity ** 2
								* np.exp(2 * self.__wave_num * z_0t))
		return u, w

	def analytical_drift_velocity(self, x_0=0, z_0=0, t=0):
		bprime = 1 - self.__beta

		# equation (13)
		u_d = self.__max_velocity ** 2 / self.__phase_velocity \
				  * (1 - self.__beta * bprime * self.__stokes_num ** 2) \
				  * np.exp(2 * self.__wave_num * (z_0 - bprime * self.__gravity 
				  * self.__st_response_time * t))
#		delta = 40 / self.__angular_freq	# shift anomaly in w_d
		# equation (14)
		w_d = -bprime * self.__gravity * self.__st_response_time \
				  - 2 * bprime * self.__stokes_num \
				  * self.__max_velocity ** 2 / self.__phase_velocity \
				  * np.exp(2 * self.__wave_num * (z_0 - bprime * self.__gravity
							 * self.__st_response_time * t))
#							 * self.__st_response_time * (t - delta)))
		return u_d, w_d

	def my_analytical_drift_velocity(self, x_0=0, z_0=0, z=0, u=0, w=0, t=0):
		bprime = 1 - self.__beta
		e_2z0t = np.exp(2 * (self.__wave_num * z_0 - self.__stokes_num * bprime
											 * t * self.__angular_freq))

		# equation (13) derived using my equations (6) and (7)
		u_d = e_2z0t * (1 - bprime * self.__stokes_num ** 2) \
					 * (self.__max_velocity ** 2 / self.__phase_velocity)

		# equation (14) derived using my equations (6) and (7)
		w_d = -self.__phase_velocity * self.__stokes_num * bprime \
				* (1 + 2 * self.__max_velocity ** 2 
					 / self.__phase_velocity ** 2 * e_2z0t)

		return u_d, w_d

	def particle_trajectory(self, x_0=0, z_0=0, delta_t=5e-3, method='BDF'):
		num_periods = 50
		num_steps = int(np.rint(num_periods * self.__period / delta_t))
		t_span = (0, num_periods * self.__period)
		t_eval = np.linspace(0, num_periods * self.__period, num_steps)
		u_0, w_0 = self.analytical_particle_velocity()
		sols = integrate.solve_ivp(self.model, t_span, [x_0, z_0, u_0, w_0],
								   method=method, t_eval=t_eval, rtol=1e-8,
								   atol=1e-10)
		x, z, u, w = sols.y
		t = sols.t
		return x, z, u, w, t

	def analytical_period_info(self, x, z, t):
		# TODO find t's where analytical u = 0 at x_0, z_0 = 0, 0
		# TODO save into list period_end_t
		# TODO make list of zeros called period_end_u the same length
		# TODO compute w's at each t, save into period_end_w list

		u, w = self.analytical_particle_velocity(t=t)
		period_end = []
		for i in range(1, len(u)):
			if u[i - 1] < 0 and 0 <= u[i]:
				period_end.append(i)

		current_x, current_z, current_u, current_w, current_t = [], [], [], \
															    [], []
		for i in range(1, len(period_end)):
			current = period_end[i]
			current_x.append(x[current])
			current_z.append(z[current])
			current_u.append(u[current])
			current_w.append(w[current])
			current_t.append(t[current])

		return period_end, np.array(current_x), np.array(current_z), \
			   np.array(current_u), np.array(current_w), np.array(current_t)

	def analytical_averages(self, x, z, u, w, t):
		averaged_u, averaged_w = [], []
		period_end, _, _, _, _, period_t = self.analytical_period_info(x, z, t)
		analytical_u, analytical_w = self.analytical_particle_velocity(t=t)

		for i in range(1, len(period_end)):
			previous = period_end[i - 1]
			current = period_end[i]
			averaged_u.append(np.average(analytical_u[previous:current]))
			averaged_w.append(np.average(analytical_w[previous:current]))

		return np.array(averaged_u), np.array(averaged_w), np.array(period_t)

	def interpolated_period_info(self, x, z, u, w, t):
		period_end = []
		for i in range(1, len(u)):
			if u[i - 1] < 0 and 0 <= u[i]:
				period_end.append(i)

		previous_t, current_t, previous_u, current_u, previous_w, current_w, \
			interpd_t, interpd_u, interpd_w, \
			interpd_x, interpd_z = [], [], [], [], [], [], [], [], [], [], []

		for i in range(1, len(period_end)):
			current = period_end[i]
			previous = current - 1

			previous_t.append(t[previous])
			previous_u.append(u[previous])
			previous_w.append(w[previous])

			current_t.append(t[current])
			current_u.append(u[current])
			current_w.append(w[current])

			new_t = np.interp(0, [u[previous], u[current]],
							  [t[previous], t[current]])
			interpd_t.append(new_t)
			interpd_u.append(0)
			interpd_w.append(np.interp(new_t, [t[previous], t[current]],
							 [w[previous], w[current]]))
			interpd_x.append(np.interp(new_t, [t[previous], t[current]],
							 [x[previous], x[current]]))
			interpd_z.append(np.interp(new_t, [t[previous], t[current]],
							 [z[previous], z[current]]))

		return np.array(previous_t), np.array(current_t), np.array(previous_u),\
			   np.array(current_u), np.array(previous_w), \
			   np.array(current_w), np.array(interpd_t), np.array(interpd_u), \
			   np.array(interpd_w), np.array(interpd_x), np.array(interpd_z)

	def numerical_drift_velocity(self, x, z, u, w, t, x_0=0, z_0=0):
		numerical_u, numerical_w = [], []
		previous_t, current_t, previous_u, current_u, previous_w, current_w, \
			interpd_t, interpd_u, interpd_w, \
			interpd_x, interpd_z = self.interpolated_period_info(x, z, u, w, t)

		for i in range(1, len(interpd_t)):
			numerical_u.append((interpd_x[i] - interpd_x[i - 1]) \
							   / (interpd_t[i] - interpd_t[i - 1]))
			numerical_w.append((interpd_z[i] - interpd_z[i - 1]) \
							   / (interpd_t[i] - interpd_t[i - 1]))

		return np.array(numerical_u), np.array(numerical_w), \
			   np.array(interpd_t[1:])

	def model(self, t, y):
		x, z = y[:2]
		particle_velocity = y[2:]
		fluid_velocity = self.fluid_velocity(x, z, t)
		fluid_derivative = self.fluid_derivative(x, z, t)

		stokes_drag = (fluid_velocity - particle_velocity) \
					  / self.__st_response_time
		buoyancy_force = (1 - self.__beta) * np.array([0, -self.__gravity]) 
		fluid_pressure_gradient = self.__beta * fluid_derivative
		particle_accel = stokes_drag + buoyancy_force + fluid_pressure_gradient

		return np.concatenate((particle_velocity, particle_accel))

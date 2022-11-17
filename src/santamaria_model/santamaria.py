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
		fluid_velocity = self.fluid_velocity(x, z, t)
		return np.array([self.__angular_freq * self.__max_velocity
						* np.exp(self.__wave_num * z)
						* np.sin(self.__wave_num * x - self.__angular_freq * t),
						-self.__angular_freq * self.__max_velocity
						* np.exp(self.__wave_num * z)
						* np.cos(self.__wave_num * x - self.__angular_freq * t)]
						+ np.dot(fluid_velocity, np.gradient(fluid_velocity)))

	def analytical_velocities(self, x_0=0, z_0=0, t=0):
		# local variables for equations (11) and (12)
		bprime = 1 - self.__beta
		phi = self.__wave_num * x_0 - self.__angular_freq * t
		z_0t = z_0 - self.__stokes_num * bprime * t

		# equation (11)
		analytical_u = self.__max_velocity * np.exp(self.__wave_num * z_0t) \
											 * ((1 - self.__stokes_num ** 2
												   * self.__beta * bprime)
												   * np.cos(phi)
												   - self.__stokes_num * bprime
												   * np.sin(phi)) \
											 + self.__max_velocity ** 2 \
											 / self.__phase_velocity \
											 * np.exp(2 * self.__wave_num
														* z_0t) \
											 * (1 - self.__stokes_num ** 2
												  * self.__beta * bprime)
		# equation (12)
		analytical_w = self.__max_velocity * np.exp(self.__wave_num * z_0t) \
										   * ((1 - self.__stokes_num ** 2
												 * self.__beta * bprime)
												 * np.sin(phi)
												 + self.__stokes_num * bprime
												 * np.cos(phi)) \
										   - self.__phase_velocity \
										   * self.__stokes_num * bprime \
										   * (1 + 2 * self.__max_velocity ** 2
												/ self.__phase_velocity ** 2
												* np.exp(2 * self.__wave_num
														   * z_0t))
		return analytical_u, analytical_w

	def particle_trajectory(self, x_0, z_0, method='BDF'):
		t_span = (0, 200 * self.__period)
		t_eval = np.linspace(0, 50 * self.__period, 7500)
		u_0, w_0 = self.analytical_velocities()
		sols = integrate.solve_ivp(self.model, t_span, [x_0, z_0, u_0, w_0],
								   method=method, t_eval=t_eval, rtol=1e-8,
								   atol=1e-10)
		x, z, u, w = sols.y
		t = sols.t
		return x, z, u, w, t

	def period_info(self, x, z, u, w, t):
#		u, w = self.analytical_velocities(t=t)
		period_end = []
		for i in range(1, len(u)):
			if u[i - 1] < 0 and 0 <= u[i]:
				period_end.append(i)

#		return period_end

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

		return previous_t, current_t, previous_u, current_u, previous_w, \
			   current_w, interpd_t, interpd_u, interpd_w, interpd_x, interpd_z

	def drift_velocities(self, x, z, u, w, t, x_0=0, z_0=0):
		numerical_u, numerical_w, numerical_t = [], [], []
		previous_t, current_t, previous_u, current_u, previous_w, current_w, \
			interpd_t, interpd_u, interpd_w, \
			interpd_x, interpd_z = self.period_info(x, z, u, w, t)
		analytical_u, analytical_w = self.analytical_velocities(t=t)
#		period_end = self.period_info(x, z, u, w, t)

		for i in range(1, len(interpd_t)):
#		for i in range(1, len(period_end)):
			previous = period_end[i - 1]
			current = period_end[i]
			numerical_u.append((interpd_x[i] - interpd_x[i - 1]) \
							   / (interpd_t[i] - interpd_t[i - 1]))
			numerical_w.append((interpd_z[i] - interpd_z[i - 1]) \
							   / (interpd_t[i] - interpd_t[i - 1]))
#			numerical_u.append(np.average(analytical_u[previous:current]))
#			numerical_w.append(np.average(analytical_w[previous:current]))
			numerical_t.append(interpd_t[i])
#			numerical_t.append(t[current])

		# pinpointing where the problematic value arises
		high_w = [num for num in numerical_w \
				  if num / self.__max_velocity > -0.123][0]
		print(f'%-15s%-10s' % ('u_d', 't'))
		for i in range(len(numerical_w)):
			if numerical_w[i] == high_w:
				print(f'%-15f%-10f ***' % (numerical_w[i], numerical_t[i]))
			else:
				print(f'%-15f%-10f' % (numerical_w[i], numerical_t[i]))

		# curve fitting to w as an attempt to compute the avg using integration
#		f = lambda x, a, b : np.exp(a * x) * np.cos(b * x)
#		a, b = curve_fit(f, t, w)[0]

#		numerical_w, numerical_t = [], []
#		period = 2 * np.pi / self.__angular_freq

#		test_result = integrate.quad(f, 0, period, args=(a, b,))[0]
#		print(test_result)

#		for num in range(15):
#			c = num * period
#			d = c + period
#			numerical_w.append(integrate.quad(f, c, d, args=(a, b,))[0]
#							   / period)
#			numerical_t.append(d)

		# equation (13)
		analytical_ud = self.__max_velocity ** 2 / self.__phase_velocity \
					   * (1 - self.__beta * (1 - self.__beta)
							* self.__stokes_num ** 2) \
					   * np.exp(2 * self.__wave_num * (z_0 - (1 - self.__beta)
								  * self.__gravity * self.__st_response_time
								  * t))
		delta = 40 / self.__angular_freq	# shift anomaly in w_d
		# equation (14)
		analytical_wd = -(1 - self.__beta) * self.__gravity \
					   * self.__st_response_time - 2 * (1 - self.__beta) \
					   * self.__stokes_num * self.__max_velocity ** 2 \
					   / self.__phase_velocity * np.exp(2 * self.__wave_num
					   * (z_0 - (1 - self.__beta) * self.__gravity
					   * self.__st_response_time * t))
#					   * self.__st_response_time * (t - delta)))

		# trying to find values of u near omega * t = 100
#		near_100 = self.__angular_freq * np.array(numerical_t) - 100
#		near_100_results = (np.abs(near_100) < 5).nonzero()[0]
#		print('numerical u_d near omega * t = 100')
#		print(f'%-22s%-25s' % ('u_d / U', 'omega * t'))
#		for i in near_100_results:
#			print(f'%-22f%-25f' % (numerical_u[i] / self.__max_velocity,
#								 self.__angular_freq * numerical_t[i]))
#		numerical_result = numerical_u[near_100_results[0]]

#		near_100 = self.__angular_freq * np.array(t) - 100
#		near_100_results = (np.abs(near_100) < 1).nonzero()[0]
#		print('\nanalytical u_d near omega * t = 100')
#		print(f'%-22s%-25s' % ('u_d / U', 'omega * t'))
#		for i in near_100_results:
#			print(f'%-22f%-25f' % (analytical_u[i] / self.__max_velocity,
#								 self.__angular_freq * t[i]))
#		analytical_result = analytical_u[(np.abs(self.__angular_freq * t - numerical_result)).argmin()]
#		print(f'\ndifference between closest numerical and analytical u_d = %f\n'
#			  % (np.abs(numerical_result - analytical_result)))

		return numerical_u, numerical_w, numerical_t, analytical_ud, \
			   analytical_wd, analytical_u, analytical_w

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

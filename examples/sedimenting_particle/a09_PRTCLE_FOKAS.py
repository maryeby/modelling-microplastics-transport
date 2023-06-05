import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport/examples/'
				+ 'sedimenting_particle')
import numpy as np
import numpy.polynomial.chebyshev as cheb
import time
from scipy.integrate import quad
from scipy.optimize import newton, newton_krylov, fsolve, broyden1
from scipy.sparse.linalg import spsolve
from os.path import exists
from progressbar import progressbar
from a00_PMTERS_CONST import MaxeyRileyParameter

class MaxeyRileyFokas(object):
	"""
	Calculates the trajectory and velocity of a particle whose dynamics are
	governed by the Maxey-Riley equation, using the Fokas method described in
	Prasath et al. (2019).
	"""

	def __init__(self, tag, x, v, velocity_field, Nz, t0, dt, time_nodes,
		 		 particle_density=1, fluid_density=1, particle_radius=1,
				 kinematic_viscosity=1, time_scale=1):
		"""
		Attributes
		----------
		tag	: str or int
			Particle name or tag.
		x : array
			The initial position of the particle.
		v : array
			The initial velocity of the particle.
		velocity_field : VelocityField (obj)
			The velocity field through which the particle is travelling.
		Nz : int
			The number of nodes in the frequency domain.
		t0 : float
			The initial time.
		dt : float
			The time step delta t.
		time_nodes : int
			The number of nodes in each time step.
		particle_density : float, default=1
			The density of the particle, rho_p.
		fluid_density : float, default=1
			The density of the fluid, rho_f.
		particle_radius : float, default=1
			The radius of the particle *a*.
		kinematic_viscosity : float, default=1
			The kinematic viscosity *nu*.
		time_scale : float, default=1
			The time scale of the flow.
		"""
		self.tag = tag
		self.t0 = t0
		self.time = t0
		self.dt = dt
		self.x = x[0]
		self.y = x[1]
		self.vel = velocity_field
		u0, v0 = velocity_field.get_velocity(x[0], x[1], t0)
	  
		self.qx_tj = np.zeros([1, Nz])[0]
		self.qx_tj[0] = v[0] - u0	# Initial condition at subdomain
		self.qy_tj = np.zeros([1, Nz])[0]
		self.qy_tj[0] = v[1] - v0	# Initial condition at subdomain
	  
		self.q0 = np.array([self.qx_tj[0], self.qy_tj[0]])
		self.p = MaxeyRileyParameter(particle_density, fluid_density,
									 particle_radius, kinematic_viscosity,
									 time_scale)
		self.time_nodes = time_nodes
	  
		index_v	= np.arange(0, Nz)
		self.z_hat_v = (1 - np.cos(index_v * np.pi / Nz)) - 1
		self.z_v = (1 + self.z_hat_v) / (1 - self.z_hat_v)
	  
		#self.k_v = np.arange(0.0, 200.0, 1e-2) 
		self.k_hat_v = (1 - np.cos(index_v * np.pi / Nz)) - 1
		self.k_v = (1 + self.k_hat_v) / (1 - self.k_hat_v)
	  
		self.pos_vec = np.copy(x)
		self.q_vec = np.copy(self.q0)
	  
		if self.vel.limits == True:
			if (self.x[0] > self.vel.x_right or self.x[0] < self.vel.x_left \
				or self.x[1] > self.vel.y_up or self.x[1] < self.vel.y_down):
				raise Exception('Particle\'s initial position is outside the '
								+ 'spatial domain.') 
	  
		self.qx_x0 = np.zeros([1, time_nodes])[0]
		self.qx_x0[0] = self.q0[0] #v[0] - u0
		self.qy_x0 = np.zeros([1, time_nodes])[0]
		self.qy_x0[0] = self.q0[1] #v[1] - v0
	  
		self.interval_def()
		self.M_nn()
	  
	def interval_def(self):
		"""
		Defines chebyshev nodes in time subdomain [t_i, ti + dt], 
		i.e. s_1, s_2, ...
		"""
		self.time_vec = np.array([])
		x = np.array([])
		y = np.array([])
		for jj in range(0, self.time_nodes):
			self.time_vec  = np.append(self.time_vec, self.time + 0.5 \
									   * self.dt * (1 - np.cos(jj * np.pi 
									   / (self.time_nodes - 1))))
			if self.time - self.t0 == 0:
				x = np.append(x, self.x)
				y = np.append(y, self.y)
			else:
				x = np.append(x, self.x[-1])
				y = np.append(y, self.y[-1])
		self.x0_v = np.append(x, y)
	
	def Lm(self, m):
		r"""
		Calculates L(m) function,
		$$\mathcal{L}(m) = \int_0^{\infty}
		\frac{k^2 \gamma e^{-k^2 m}}{k^2 \gamma^2 + (k^2 - \alpha)^2} dk,$$
		used to obtain matrix M and then F.
		"""	
		fun_exp = lambda k: np.exp(-m * k ** 2)
		fun_frac = lambda k: self.p.gamma * k ** 2 / \
				   ((k * self.p.gamma) ** 2 + (k ** 2 - self.p.alpha) ** 2)	
		fun = lambda k: fun_exp(k) * fun_frac(k)
		fun_v = np.array([])

		for kk in range(0, len(self.k_v)):
			fun_v = np.append(fun_v, fun(self.k_v[kk]))
		
		coeff = cheb.chebfit(self.k_hat_v, fun_v, len(self.k_v) - 1)
		coeff_int = cheb.chebint(coeff)
		result = cheb.chebval(1, coeff_int) - cheb.chebval(-1, coeff_int)
		return result
	
	def M_nn(self):
		"""Calculates matrix M."""
		name_file = 'a00_MATRX_VALUES.txt'
		
		if exists(name_file) == True:
			with open(name_file, 'rb') as file:
				mat = np.load(file)
		
		if exists(name_file) == False or mat.shape[0] != len(self.time_vec):
			#print('Creating Matrix.')
			time.sleep(0.3)
			#time1   = time.time()
			mat	 = np.zeros([len(self.time_vec),len(self.time_vec)])
		
			for ii in progressbar(range(1,len(self.time_vec))):
				for nn in range(0,len(self.time_vec)):
					# Create vector of coeffs to define Chebyshev polynomial
					coeff	  = np.zeros([1,len(self.time_vec)])[0]

					# Fill in the element of the vector corresponding to the
					# matrix entry
					coeff[nn]  = 1
					poly = cheb.Chebyshev(coeff) # change into Cheb polynomial

					# Create function of full integrand of the matrix element
					fun	= lambda s: poly(s) * self.Lm(self.time_vec[ii]-s)
					aux = quad(fun, self.time_vec[0], self.time_vec[ii],
							   epsrel=1e-9, epsabs=1e-9, limit=1000,
							   points=[self.time_vec[ii]]) # integrate
					mat[ii][nn] = aux[0] # fill in matrix

					# Display values
					#print('position [t_',ii,', T_',nn,']:',mat[ii][nn])
					#print('error:',aux[1])
		
			with open(name_file, 'wb') as file:
				np.save(file, mat)
			
		self.M = mat
		#time2 = time.time()
		#print("Time to calculate Matrix: " + str(time2-time1))

	def calculate_f(self):
		"""
		Define forcing function f(q(0, t)). This will differ for different 
		boundary problems.
		"""
		return 0, self.p.sigma
  
	def F(self, q_guess, x_guess):
		"""Calculates F(t_j)."""	
		# Change solution guess according to function f
		f_vec	  = np.array([])
		g_vec	  = np.array([])
		
		for tt in range(0,len(self.time_vec)):
			q_vec  = np.array([q_guess[tt],
							   q_guess[tt + int(len(q_guess) / 2)]])
			x_vec  = np.array([x_guess[tt],
							   x_guess[tt + int(len(x_guess) / 2)]])
			f, g   = self.calculate_f()
			f_vec  = np.append(f_vec, f)
			g_vec  = np.append(g_vec, g)
		
		self.f_vec = f_vec
		self.g_vec = g_vec
		
		# Approximate by Chebyshev coeff
		coeff_x	= cheb.chebfit(self.time_vec-self.time_vec[0],
							   f_vec, len(self.time_vec)-1)
		coeff_y	= cheb.chebfit(self.time_vec-self.time_vec[0],
							   g_vec, len(self.time_vec)-1)
		
		# Obtain F as the dot product of the coeff times the matrix entries
		result_x   = np.array([])
		result_y   = np.array([])
		for jj in range(0,len(self.time_vec)):
			result_x  = np.append(result_x, (2.0/np.pi) \
								  * np.dot(coeff_x, self.M[jj]))
			result_y  = np.append(result_y, (2.0/np.pi) \
								  * np.dot(coeff_y, self.M[jj]))
		
		result_v = np.append(result_x, result_y)
		return result_v

	def q_hat(self, k):
		"""Calculates q hat"""	   
		intgnd_x = np.exp(1j * k * self.z_v) * self.qx_tj * 2 \
							 / (1 - self.z_hat_v) ** 2
		intgnd_y = np.exp(1j * k * self.z_v) * self.qy_tj * 2 \
							 / (1 - self.z_hat_v) ** 2
		
		coeff_x	= cheb.chebfit(self.z_hat_v, intgnd_x, len(self.z_hat_v) - 1)
		coeff_y	= cheb.chebfit(self.z_hat_v, intgnd_y, len(self.z_hat_v) - 1)
		coeffx_int = cheb.chebint(coeff_x)
		coeffy_int = cheb.chebint(coeff_y)	
		result_x = cheb.chebval(1, coeffx_int) - cheb.chebval(-1, coeffx_int)
		result_y = cheb.chebval(1, coeffy_int) - cheb.chebval(-1, coeffy_int)
		result	 = np.array([[result_x], [result_y]])
		return result

	def G_fun(self):
		"""Calculates G."""	
		resultx_v = np.array([])
		resulty_v = np.array([])

		for k in self.k_v:
			q_hat = self.q_hat(k)
			fun_num_x  = (1j * k * self.p.gamma - (k ** 2 - self.p.alpha)) \
							 * (self.p.gamma * q_hat[0] + self.qx_x0[0])
			fun_num_y  = (1j * k * self.p.gamma - (k ** 2 - self.p.alpha)) \
							 * (self.p.gamma * q_hat[1] + self.qy_x0[0])
			fun_den	= - (k * self.p.gamma) ** 2 - (k ** 2 - self.p.alpha) ** 2
			resultx_v  = np.append(resultx_v, fun_num_x / fun_den)
			resulty_v  = np.append(resulty_v, fun_num_y / fun_den)
		
		result_v = np.array([resultx_v, resulty_v])		
		self.G_v_imag = result_v.imag
		return result_v.imag
	
	def G_update(self):
		"""Updates the term G."""
		#f_vec = np.array([[],[]])
		f_tld_x_v = np.array([])
		f_tld_y_v = np.array([])
		self.fx_vector = np.array([])
		self.fy_vector = np.array([])
		
		for k in self.k_v:
			# Approximate by Chebyshev coeff
			exponent = -(self.time_old_v[-1] - self.time_old_v) * k ** 2
			exponent[exponent < -1e2] = -100.0
			
			coeff_x	= cheb.chebfit(self.time_old_v,
								   np.exp(exponent) * self.f_vec,
								   len(self.time_old_v) - 1)
			coeff_y	= cheb.chebfit(self.time_old_v,
								   np.exp(exponent) * self.g_vec,
								   len(self.time_old_v) - 1)

			coeffx_int = cheb.chebint(coeff_x)
			coeffy_int = cheb.chebint(coeff_y)

			f_tld_x	= cheb.chebval(self.time_old_v[-1], coeffx_int) \
					  - cheb.chebval(self.time_old_v[0], coeffx_int)
			f_tld_y	= cheb.chebval(self.time_old_v[-1], coeffy_int) \
					  - cheb.chebval(self.time_old_v[0], coeffy_int)
			
			self.fx_vector = np.append(self.fx_vector, f_tld_x)
			self.fy_vector = np.append(self.fy_vector, f_tld_y)
			
			f_tld_x_frac = k * self.p.gamma * f_tld_x \
							 / ((k * self.p.gamma) ** 2
							  + (k ** 2 - self.p.alpha) ** 2)
			f_tld_y_frac = k * self.p.gamma * f_tld_y \
							 / ((k * self.p.gamma) ** 2 
							 + (k ** 2 - self.p.alpha) ** 2)
			
			f_tld_x_v = np.append(f_tld_x_v, f_tld_x_frac)
			f_tld_y_v = np.append(f_tld_y_v, f_tld_y_frac)
			
		self.f_tld_x = f_tld_x_v
		self.f_tld_y = f_tld_y_v

		exponentx = -self.k_v ** 2 * self.dt
		exponentx[exponentx <-1e2] = -100
		resultx_v = np.exp(exponentx) * self.G_v_imag[0] - f_tld_x_v		

		exponenty = -self.k_v ** 2 * self.dt
		exponenty[exponenty <-1e2] = -100
		resulty_v = np.exp(exponenty) * self.G_v_imag[1] - f_tld_y_v
		
		resultx_v[abs(resultx_v) < 1e-15] = 0
		resulty_v[abs(resulty_v) < 1e-15] = 0
		result_v = np.array([resultx_v, resulty_v])		
		self.G_v_imag = result_v
		return result_v
	
	def H(self):
		"""Calculates H."""
		if self.time_vec[0] - self.t0 == 0:
			self.G_fun()
		else:
			self.G_update()
		resultx_v = np.array([])
		resulty_v = np.array([])
		
		for jj in range(1, len(self.time_vec)):		
			exponent = np.log((2 / np.pi) * 2 / (1 - self.k_hat_v) ** 2) \
					   - (self.time_vec[jj] - self.time_vec[0]) * self.k_v ** 2
			exponent[exponent < -1e2] = -100

			Hx_intgrnd = -np.exp(exponent) * self.G_v_imag[0] * self.k_v
			Hy_intgrnd = -np.exp(exponent) * self.G_v_imag[1] * self.k_v
			Hx_intgrnd[abs(Hx_intgrnd) < 1e-40] = 0
			Hy_intgrnd[abs(Hy_intgrnd) < 1e-40] = 0

			coeff_x	= cheb.chebfit(self.k_hat_v, Hx_intgrnd,
								   len(self.k_hat_v) - 1)
			coeff_y	= cheb.chebfit(self.k_hat_v, Hy_intgrnd,
								   len(self.k_hat_v) - 1)
			coeffx_int = cheb.chebint(coeff_x)
			coeffy_int = cheb.chebint(coeff_y)

			resultx_v = np.append(resultx_v, cheb.chebval(1, coeffx_int) \
						 - cheb.chebval(-1, coeffx_int))
			resulty_v = np.append(resulty_v, cheb.chebval(1, coeffy_int) \
						 - cheb.chebval(-1, coeffy_int))
			
		if self.time_vec[0] - self.t0 == 0:
			resultx_v = np.append(self.qx_x0[0], resultx_v)
			resulty_v = np.append(self.qy_x0[0], resulty_v)
		else:
			resultx_v = np.append(self.qx_x0[-1], resultx_v)
			resulty_v = np.append(self.qy_x0[-1], resulty_v)
		
		result_v   = np.append(resultx_v, resulty_v)
		self.H_v   = result_v
		return resultx_v, resulty_v
	
	def eta(self, q_guess):
		"""Calculates eta."""
		q_v	= np.array([])
		p_v	= np.array([])
		for tt in range(0, len(self.time_vec)):
			q_v = np.append(q_v, q_guess[tt])
			p_v = np.append(p_v, q_guess[tt+int(len(q_guess)/2)])
		
		coeff_x	= cheb.chebfit(self.time_vec, q_v, len(self.time_vec)-1)
		coeff_y	= cheb.chebfit(self.time_vec, p_v, len(self.time_vec)-1)
		intcoeff_x = cheb.chebint(coeff_x)
		intcoeff_y = cheb.chebint(coeff_y)
		value_x_0 = cheb.chebval(self.time_vec[0], intcoeff_x)
		value_y_0 = cheb.chebval(self.time_vec[0], intcoeff_y)
		
		resultx_v = np.array([0])
		resulty_v = np.array([0])
		for tt in range(1, len(self.time_vec)):
			resultx_v = np.append(resultx_v, cheb.chebval(self.time_vec[tt],
								  intcoeff_x) - value_x_0)
			resulty_v = np.append(resulty_v, cheb.chebval(self.time_vec[tt],
								  intcoeff_y) - value_y_0)
		
		result_v   = np.append(resultx_v, resulty_v)
		return result_v
	
	def Psi(self, x_guess):
		"""Calculates psi."""
		U_v = np.array([])
		V_v	= np.array([])
		for tt in range(0, len(self.time_vec)):
			x = x_guess[tt]
			y = x_guess[tt + int(len(x_guess)/2)]
			u, v = self.vel.get_velocity(x, y, self.time_vec[tt])
			U_v = np.append(U_v, u)
			V_v = np.append(V_v, v)
		
		coeff_x	= cheb.chebfit(self.time_vec, U_v, len(self.time_vec)-1)
		coeff_y	= cheb.chebfit(self.time_vec, V_v, len(self.time_vec)-1)
		intcoeff_x = cheb.chebint(coeff_x)
		intcoeff_y = cheb.chebint(coeff_y)
		value_x_0 = cheb.chebval(self.time_vec[0], intcoeff_x)
		value_y_0 = cheb.chebval(self.time_vec[0], intcoeff_y)
		
		resultx_v = np.array([0])
		resulty_v = np.array([0])
		for tt in range(1, len(self.time_vec)):
			resultx_v = np.append(resultx_v, cheb.chebval(self.time_vec[tt],
								  intcoeff_x) - value_x_0)
			resulty_v = np.append(resulty_v, cheb.chebval(self.time_vec[tt],
								  intcoeff_y) - value_y_0)
		
		result_v = np.append(resultx_v, resulty_v)
		return result_v
	
	def J(self,guess):
		"""Defines the function J to obtain rules on."""
		q_guess	= guess[:len(self.time_vec)*2]
		x_guess	= guess[len(self.time_vec)*2:]
		
		# Define function J, on which Newton method is used.
		J_v	= q_guess - self.F(q_guess, x_guess) - self.H_v
		Psi_v = self.x0_v - x_guess + self.eta(q_guess) + self.Psi(x_guess)	
		result_v = np.append(J_v, Psi_v)
		return result_v
	
	# Function that runs nonlinear solver and calculates solution at boundary.
	def update(self):
		"""Updates the velocity and trajectory of the particle."""	
		q_guess, p_guess = self.H()	# guess for the relative velocity
		
		if self.time - self.t0 == 0:
			x_guess = np.array([self.x])
			y_guess = np.array([self.y])
		else:
			x_guess = np.array([self.x[-1]])
			y_guess = np.array([self.y[-1]])
		
		# guess for the position
		for tt in range(0, len(self.time_vec)-1):
			dt = self.time_vec[tt+1] - self.time_vec[tt]
			q, p = q_guess[tt], p_guess[tt]
			u, v = self.vel.get_velocity(x_guess[-1], y_guess[-1],
										 self.time_vec[tt])
			x_guess = np.append(x_guess, x_guess[-1] + dt * (q + u))
			y_guess = np.append(y_guess, y_guess[-1] + dt * (p + v))
		
		vel_guess = np.append(q_guess, p_guess)
		pos_guess = np.append(x_guess, y_guess)
		guess = np.append(vel_guess, pos_guess)
		
		# Newton-Raphson method
		#print("Initial guess: \n" + str(guess))
		iter_limit = 1000
		tolerance  = 1e-13
		
		try:
			result = newton(self.J, guess, maxiter=iter_limit, tol=tolerance,
							full_output=True)

			# Check method converged before reaching maxiter.
			if any(np.invert(result[1])) == False: # Method converged!
				solution = result[0]
			else:
				raise
		except:
			try: # Try Newton method with Krylov inverse Jacobian approximation.
				solution = newton_krylov(self.J, guess, maxiter=iter_limit,
										 f_tol=tolerance)
			# If everything failed, use fsolve, which is slower but more stable.
			except:
				solution = fsolve(self.J, guess, maxfev=iter_limit,
								  xtol=tolerance)
		
		#print("G: " + str(self.G_v_imag))
		self.q_x0 = np.copy(solution)
		self.qx_x0 = np.copy(solution[:len(self.time_vec)])
		self.qy_x0 = np.copy(solution[len(self.time_vec): 2*len(self.time_vec)])
		self.x = np.copy(solution[2 * len(self.time_vec): 3*len(self.time_vec)])
		self.y = np.copy(solution[3 * len(self.time_vec):])
		
		#print("2 norm: \n" + str(np.linalg.norm(self.J(self.q_x0))))
		#print("Inf norm: \n" + str(np.linalg.norm(self.J(self.q_x0),np.inf)))
		#print("Method's convergence: \n" + str(result[1]))
		#print("Solution: \nq_1: "+str(self.qx_x0)+"\nq_2: "+str(self.qy_x0))
		#assert (not any(np.invert(result[1]))), 'Method does not converge!'
		
		pos_vec = np.array([self.x[1:], self.y[1:]])
		q_vec = np.array([self.qx_x0[1:], self.qy_x0[1:]])
		self.pos_vec = np.vstack([self.pos_vec, np.transpose(pos_vec)])
		self.q_vec = np.vstack([self.q_vec, np.transpose(q_vec)])
		
		self.time_old_v = self.time_vec
		self.time += self.dt
		self.interval_def()
		
		return solution

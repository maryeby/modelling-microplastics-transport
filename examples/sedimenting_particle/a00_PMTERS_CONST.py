import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport/examples/'
				+ 'sedimenting_particle')
import numpy as np

class MaxeyRileyParameter(object):
	"""
	Defines all the parameters needed for solving the MRE. The names of the 
	parameters are kept as per Prasath et al. (2019).
	"""

	def __init__(self, particle_density, fluid_density, particle_radius,
				 kinematic_viscosity, time_scale):
		"""
		Attributes
		----------
		particle_density : float
			The density of the particle, rho_p.
		fluid_density : float
			The density of the fluid, rho_f.
		particle_radius : float
			The radius of the particle *a*.
		kinematic_viscosity : float
			The kinematic viscosity nu.
		time_scale : float
			The time scale *T*.
		beta : float
			The ratio between the particle and fluid densities.
		S : float
			The Stokes number.
		R : float
			Another density ratio, related to beta.
		alpha : float
			The coefficient of the Stokes drag term in the reformulation.
		gamma : float
			The coefficient of the history term in the reformulation.
		sigma : float
		"""
		self.rho_p = particle_density
		self.rho_f = fluid_density
		self.a = particle_radius
		self.nu	= kinematic_viscosity
		self.T = time_scale

		self.set_beta()
		self.set_S()
		self.set_R()
		self.set_alpha()
		self.set_gamma()
		self.set_sigma()

	def set_beta(self):
		"""Defines beta, the ratio of the particle and fluid densities."""
		self.beta = self.rho_p / self.rho_f
  
	def set_S(self):
		"""Defines the Stokes number *S*."""
		self.S = (1 / 3) * self.a ** 2 / (self.nu * self.T)
  
	def set_R(self):
		"""Defines *R*, a parameter related to beta."""
		self.R = (1 + 2 * self.beta) / 3
	
	def set_alpha(self):
		"""Defines alpha, used in the reformulation."""
		self.alpha = 1 / (self.R * self.S) 

	def set_gamma(self):
		"""Defines gamma, used in the reformulation."""
		self.gamma = (1 / self.R) * np.sqrt(3 / self.S)

	def set_sigma(self):
		"""Defines sigma, the non-dimensional parameter for gravity."""
		self.sigma = 1 / self.R - 1 

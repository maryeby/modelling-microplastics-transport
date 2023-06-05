import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport/examples/'
				+ 'sedimenting_particle')
from abc import ABC, abstractmethod

class VelocityField(ABC):
	"""Abstract class that defines the velocity field."""

	def __init__(self):
		"""
		Attributes
		----------
		limits : boolean
			Whether the velocity field is defined over a bounded domain.
		x_left : float
			The left x-limit.
		x_right : float
			The right x-limit.
		y_left : float
			The left y-limit.
		y_right : float
			The right y-limit.

		Notes
		-----
		Analytical velocity fields do not ususally have limits
		"""
		self.limits = False
		self.x_left  = None
		self.x_right = None
		self.y_down  = None
		self.y_up	= None
		pass
	
	@abstractmethod
	def get_velocity(self, x, y, t):
		"""
		Defines the velocity of the flow.
		
		Parameters
		----------
		x : float
			The horizontal position at which to evaluate the velocity.
		y : float
			The vertical position at which to evaluate the velocity.
		t : float
			The time at which to evaluate the velocity.

		Returns
		-------
		u : float
			The horizontal velocity.
		v : float
			The vertical velocity.
		"""
		u = 0 * x
		v = 0 * y
		return u, v
	
	@abstractmethod
	def get_gradient(self, x, y, t):
		"""
		Defines spatial derivatives of the velocity field.
		
		Parameters
		----------
		x : float
			The horizontal position at which to evaluate the derivative.
		y : float
			The vertical position at which to evaluate the derivative.
		t : float
			The time at which to evaluate the derivative.

		Returns
		-------
		ux : float
			The x derivative of u.
		uy : float
			The y derivative of u.
		vx : float
			The x derivative of v.
		vy : float
			The y derivative of v.
		"""
		ux, uy = 0 * x, 0 * y
		vx, vy = 0 * x, 0 * y
		return ux, uy, vx, vy

	@abstractmethod
	def get_dudt(self, x, y, t):
		"""
		Define time derivative of the velocity field.
		
		Parameters
		----------
		x : float
			The horizontal position at which to evaluate the derivative.
		y : float
			The vertical position at which to evaluate the derivative.
		t : float
			The time at which to evaluate the derivative.

		Returns
		-------
		ut : float
			The time derivative of u.
		vt : float
			The time derivative of v.
		"""
		ut = 0 * x
		vt = 0 * y
		return ut, vt

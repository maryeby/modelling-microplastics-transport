class Particle:
	"""Respresents a rigid, spherical, inertial particle."""

	def __init__(self, stokes_num):
		"""
		Attributes
		----------
		stokes_num : float
			The stokes number *St*.
		"""
		self.stokes_num = stokes_num

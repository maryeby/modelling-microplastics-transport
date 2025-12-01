class Particle:
	"""Respresent a rigid, spherical, inertial particle."""

	def __init__(self, stokes_hat):
		r"""
		Attributes
		----------
		stokes_hat : float
			The density-independent Stokes number $\widehat{St}$.
		"""
		self.stokes_hat = stokes_hat

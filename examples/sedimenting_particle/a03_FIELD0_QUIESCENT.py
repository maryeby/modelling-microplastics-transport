#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport/examples/'
				+ 'sedimenting_particle')
import numpy as np
from a03_FIELD0_00000 import VelocityField

class QuiescentFlow(VelocityField):
	"""Defines of the velocity field corresponding to a quiescent flow."""

	def __init__(self):
		self.limits = False # not including limits for an analytical flow
	
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
		u = 0.0
		v = 0.0
		return u, v
 
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
		ux = 0.0
		uy = 0.0
		vx = 0.0
		vy = 0.0
		return ux, uy, vx, vy

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
		# Define time derivatives of the field
		ut = 0.0
		vt = 0.0
		return ut, vt

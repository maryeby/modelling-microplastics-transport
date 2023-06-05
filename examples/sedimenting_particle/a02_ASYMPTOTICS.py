#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport/examples/'
				+ 'sedimenting_particle')
import numpy as np
import multiprocessing as mp
import pandas as pd
from progressbar import progressbar

from a03_FIELD0_QUIESCENT import QuiescentFlow
from a09_PRTCLE_FOKAS import MaxeyRileyFokas

def main():
	r"""
	Produces asymptotic solutions for a particle's motion in a fluid flow, where
	$$c(\alpha, \gamma) = \frac{\sigma}{\alpha}$$
	is the velocity of the particle at long times.
	"""
	# Define time grid
	tini = 0.0	# Initial time
	tend = 10.0	# Final time
	nt = 11
	#nt   = 351	# Time nodes, delta t approx 0.03
	#nt   = 2327	# Time nodes, delta t approx 0.0043

	# Create time axis, where S = 0.01
	taxis  = np.linspace(tini, tend, nt)
	dt = taxis[1] - taxis[0]

	# Define particle's and fluid's parameters
	rho_p = [5.0, 1.0, 0.01]	# Particle's density
	rho_f = 1.0					# Fluid's density 
	rad_p = np.sqrt(3)			# Particle's radius
	nu_f = 1.0					# Kinematic viscocity
	t_scale = 100.0				# Time scale of the flow

	# Import chosen velocity field
	vel = QuiescentFlow() # quiescent velocity field

	# Define number of nodes per time step
	nodes_dt = 20 # Nodes in each time step, as per Prasath et al. (2019)

	# Definition of the pseudo-spatial grid for Fokas
	N_fokas = 101 # Nodes in the frequency domain

	# Define particle's initial conditions
	x0, y0    = 0.0, 0.0
	u0, v0    = 1.0, 0.0

	# Create particles at various densities
	particle1 = MaxeyRileyFokas(1, np.array([x0, y0]), np.array([u0, v0]),
								vel, N_fokas, tini, dt, nodes_dt,
								particle_density=rho_p[0],
								fluid_density=rho_f,
								particle_radius=rad_p,
								kinematic_viscosity=nu_f,
								time_scale=t_scale)
	particle2 = MaxeyRileyFokas(2, np.array([x0, y0]), np.array([u0, v0]),
								vel, N_fokas, tini, dt, nodes_dt,
								particle_density=rho_p[1],
								fluid_density=rho_f,
								particle_radius=rad_p,
								kinematic_viscosity=nu_f,
								time_scale=t_scale)
	particle3 = MaxeyRileyFokas(3, np.array([x0, y0]), np.array([u0, v0]),
								vel, N_fokas, tini, dt, nodes_dt,
								particle_density=rho_p[2],
								fluid_density=rho_f,
								particle_radius=rad_p,
								kinematic_viscosity=nu_f,
								time_scale=t_scale)

	# Compute asymptotic data using (4.8) with sign corrected singular term
	stokes_num = rad_p ** 2 / (3 * t_scale * nu_f)
	sigma = np.array([particle1.p.sigma, particle2.p.sigma,
					  particle3.p.sigma])
	beta = np.zeros(len(rho_p))
	r = np.zeros(len(rho_p))
	alpha = np.zeros(len(rho_p))
	gamma = np.zeros(len(rho_p))

	for i in range(len(rho_p)):
		beta[i] = rho_p[i] / rho_f
		r[i] = (1 + 2 * beta[i]) / 3
		alpha[i] = 1 / (r[i] * stokes_num)
		gamma[i] = 1 / r[i] * np.sqrt(3 / stokes_num)	

	q0_1_asymp = sigma[0] / alpha[0]
	q0_2_asymp = sigma[1] / alpha[1]
	q0_3_asymp = sigma[2] / alpha[2]

	# Store results and write to a csv file
	asymptotics = pd.DataFrame({'q01_asymp': [1 + q0_1_asymp],
								'q02_asymp': [1 + q0_2_asymp],
								'q03_asymp': [1 + q0_3_asymp]})
	if __name__ == '__main__':
		asymptotics.to_csv('../data/asymptotic_results.csv', index=False)
		plot_labels = pd.DataFrame({'sigma': sigma, 'beta': beta})
		plot_labels.to_csv('../data/plot_labels.csv', index=False)
main()

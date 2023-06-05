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
	"""
	Produces numerical solutions for a particle's motion in a fluid flow.

	Notes
	-----
	- Based on work by author cfg4065.
	- The nonlinear solving process may not converge if two consecutive time
	  nodes are too close to each other (for example if a large amount of nodes 
	  is provided). This could be addressed by either (1) changing the 
	  nonlinear solver, (2) increasing the number of maximum iterations or (3) 
	  decreasing the tolerance. These parameters can be changed in the 
	  `a09_PRTCLE_FOKAS script`, under the `update` method.
	- The velocity field is defined in the `a03_FIELD0_QUIESCENT` file, whose
	  class is imported.
	- Changing the parameters related to the time grid, i.e. `tini`, `tend`,
	  `nt`, may require a recalculation of the matrix values to ensure
	  convergence of the nonlinear solver (automatically done by the ToolBox by 
	  deleting the `a00_MATRX_VALUES.txt` file).
	"""
	# Define time grid
	tini = 0.0	# Initial time
	tend = 10.0	# Final time
	nt = 6
	#nt = 351	# Time nodes, delta t approx 0.03
	#nt = 2327	# Time nodes, delta t approx 0.0043

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
	nodes_dt = 20 # Nodes in each time step

	# Definition of the pseudo-spatial grid for Fokas
	N_fokas = 101 # Nodes in the frequency domain

	# Decide whether to apply parallel computing
	parallel_flag = True
	number_cores  = int(mp.cpu_count())

	# Define particle's initial conditions
	x0, y0    = 0.0, 0.0
	u0, v0    = 1.0, 0.0

	# Create particles and calculate their velocities
	particle1 = MaxeyRileyFokas(1, np.array([x0, y0]), np.array([u0, v0]),
								vel, N_fokas, tini, dt, nodes_dt,
								particle_density=rho_p[0],
								fluid_density=rho_f,
								particle_radius=rad_p,
								kinematic_viscosity=nu_f,
								time_scale=t_scale)
	q0_1_vec = np.array([u0, v0])
	for tt in progressbar(range(1, len(taxis))):
		particle1.update()
		q0_1_vec = np.vstack((q0_1_vec, particle1.q_vec[tt * (nodes_dt-1)]))

	particle2 = MaxeyRileyFokas(2, np.array([x0, y0]), np.array([u0, v0]),
								vel, N_fokas, tini, dt, nodes_dt,
								particle_density=rho_p[1],
								fluid_density=rho_f,
								particle_radius=rad_p,
								kinematic_viscosity=nu_f,
								time_scale=t_scale)
	q0_2_vec = np.array([u0, v0])
	for tt in progressbar(range(1, len(taxis))):
		particle2.update()
		q0_2_vec = np.vstack((q0_2_vec, particle2.q_vec[tt * (nodes_dt-1)]))

	particle3 = MaxeyRileyFokas(3, np.array([x0, y0]), np.array([u0, v0]),
								vel, N_fokas, tini, dt, nodes_dt,
								particle_density=rho_p[2],
								fluid_density=rho_f,
								particle_radius=rad_p,
								kinematic_viscosity=nu_f,
								time_scale=t_scale)
	q0_3_vec = np.array([u0, v0])
	for tt in progressbar(range(1, len(taxis))):
		particle3.update()
		q0_3_vec = np.vstack((q0_3_vec, particle3.q_vec[tt * (nodes_dt-1)]))

	# Create particles and calculate their trajectories
	u0, v0 = 1.0, 1.0
	particle4 = MaxeyRileyFokas(4, np.array([x0, y0]), np.array([u0, v0]),
								vel, N_fokas, tini, dt, nodes_dt,
								particle_density=rho_p[0],
								fluid_density=rho_f,
								particle_radius=rad_p,
								kinematic_viscosity=nu_f,
								time_scale=t_scale)
	pos1_vec = np.array([x0, y0])
	for tt in progressbar(range(1, len(taxis))):
		particle4.update()
		pos1_vec = np.vstack((pos1_vec,
							  particle4.pos_vec[tt * (nodes_dt - 1)]))

	particle5 = MaxeyRileyFokas(5, np.array([x0, y0]), np.array([u0, v0]),
								vel, N_fokas, tini, dt, nodes_dt,
								particle_density=rho_p[1],
								fluid_density=rho_f,
								particle_radius=rad_p,
								kinematic_viscosity=nu_f,
								time_scale=t_scale)
	pos2_vec = np.array([x0, y0])
	for tt in progressbar(range(1, len(taxis))):
		particle5.update()
		pos2_vec = np.vstack((pos2_vec,
							  particle5.pos_vec[tt * (nodes_dt - 1)]))

	particle6 = MaxeyRileyFokas(6, np.array([x0, y0]), np.array([u0, v0]),
								vel, N_fokas, tini, dt, nodes_dt,
								particle_density=rho_p[2],
								fluid_density=rho_f,
								particle_radius=rad_p,
								kinematic_viscosity=nu_f,
								time_scale=t_scale)
	pos3_vec = np.array([x0, y0])
	for tt in progressbar(range(1, len(taxis))):
		particle6.update()
		pos3_vec = np.vstack((pos3_vec,
							  particle6.pos_vec[tt * (nodes_dt - 1)]))

	# Store results and write to a csv file
	numerics = pd.DataFrame({'t_numerical': taxis,
							 'q01': 1 + q0_1_vec[:, 1],
							 'q02': 1 + q0_2_vec[:, 1],
							 'q03': 1 + q0_3_vec[:, 1],
							 'x1': pos1_vec[:, 0],
							 'z1': pos1_vec[:, 1],
							 'x2': pos2_vec[:, 0],
							 'z2': pos2_vec[:, 1],
							 'x3': pos3_vec[:, 0],
							 'z3': pos3_vec[:, 1]})
	if __name__ == '__main__':
		numerics.to_csv('../data/numerical_results.csv', index=False)
main()

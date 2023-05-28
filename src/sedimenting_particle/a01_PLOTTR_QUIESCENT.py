#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('../src')
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
from progressbar import progressbar
from matplotlib import ticker

from a03_FIELD0_QUIESCENT import velocity_field_quiescent
from a09_PRTCLE_FOKAS import maxey_riley_fokas

"""
Created on Wed Sep  7 16:08:30 2022

@author: cfg4065

This script reproduces the figures from Example 4 in the paper
'Accurate Solution method for the Maxey-Riley equation, and the
effects of Basset history' by S. G. Prasath et al. (2019)

We provide plots of a particles' velocity and trajectory on a
static fluid.

Please be aware that:
    - all plots are printed as PDFs and saved into the folder
      '02_VSUAL_OUTPUT',
    - the nonlinear solving process may not converge if two
      consecutive time nodes are too close to each other (for
      example if WAY TOO MUCH of a bigger amount of nodes is
      provided). This could be addressed by either
      (1) changing the nonlinear solver, (2) increasing the
      number of maximum iterations or (3) decreasing the
      tolerance. These parameters can be changed in the
      'a09_PRTCLE_FOKAS script', under the 'update' method.
    - The velocity field is defined in the 'a03_FIELD0_DATA1'
      file, whose class is imported.
"""

# Define folder where to save data
save_plot_to    = './02_VSUAL_OUTPUT/'

# Define time grid
tini = 0.0	# Initial time
tend = 10.0	# Final time
nt   = 351	# Time nodes, delta t approx 0.03
#nt   = 2327	# Time nodes, delta t approx 0.0043

# Create time axis, where S = 0.01
taxis  = np.linspace(tini, tend, nt)
dt = taxis[1] - taxis[0]

# Define particle's and fluid's parameters
rho_p   = [5.0, 1.0, 0.01]			# Particle's density
rho_f   = 1.0						# Fluid's density 
rad_p   = np.sqrt(3)				# Particle's radius
nu_f    = 1.0						# Kinematic viscocity
t_scale = 100.0						# Time scale of the flow

# Import chosen velocity field
vel = velocity_field_quiescent() # quiescent velocity field

# Define number of nodes per time step
nodes_dt = 20 # Nodes in each time step, as per Prasath et al. (2019)

# Definition of the pseudo-spatial grid for Fokas
N_fokas = 101 # Nodes in the frequency domain, k, as per Prasath et al. (2019)

# Decide whether to apply parallel computing
parallel_flag = True
number_cores  = int(mp.cpu_count())

# Define particle's initial conditions
x0, y0    = 0.0, 0.0
u0, v0    = 1.0, 0.0

# Create particles and calculate the trajectories
particle1 = maxey_riley_fokas(1, np.array([x0, y0]),
                                 np.array([u0, v0]),
                                 vel, N_fokas, tini, dt,
                                 nodes_dt,
                                 particle_density    = rho_p[0],
                                 fluid_density       = rho_f,
                                 particle_radius     = rad_p,
                                 kinematic_viscosity = nu_f,
                                 time_scale          = t_scale)
pos1_vec = np.array([x0, y0])
q0_1_vec = np.array([u0, v0])
for tt in progressbar(range(1, len(taxis))):
    particle1.update()
    pos1_vec = np.vstack((pos1_vec, particle1.pos_vec[tt * (nodes_dt-1)]))
    q0_1_vec = np.vstack((q0_1_vec, particle1.q_vec[tt * (nodes_dt-1)]))

particle2 = maxey_riley_fokas(2, np.array([x0, y0]),
                                 np.array([u0, v0]),
                                 vel, N_fokas, tini, dt,
                                 nodes_dt,
                                 particle_density    = rho_p[1],
                                 fluid_density       = rho_f,
                                 particle_radius     = rad_p,
                                 kinematic_viscosity = nu_f,
                                 time_scale          = t_scale)
pos2_vec = np.array([x0, y0])
q0_2_vec = np.array([u0, v0])
for tt in progressbar(range(1, len(taxis))):
    particle2.update()
    pos2_vec = np.vstack((pos2_vec, particle2.pos_vec[tt * (nodes_dt-1)]))
    q0_2_vec = np.vstack((q0_2_vec, particle2.q_vec[tt * (nodes_dt-1)]))

particle3 = maxey_riley_fokas(3, np.array([x0, y0]),
                                 np.array([u0, v0]),
                                 vel, N_fokas, tini, dt,
                                 nodes_dt,
                                 particle_density    = rho_p[2],
                                 fluid_density       = rho_f,
                                 particle_radius     = rad_p,
                                 kinematic_viscosity = nu_f,
                                 time_scale          = t_scale)
pos3_vec = np.array([x0, y0])
q0_3_vec = np.array([u0, v0])
for tt in progressbar(range(1, len(taxis))):
    particle3.update()
    pos3_vec = np.vstack((pos3_vec, particle3.pos_vec[tt * (nodes_dt-1)]))
    q0_3_vec = np.vstack((q0_3_vec, particle3.q_vec[tt * (nodes_dt-1)]))

# Read data extracted from Prasath et al. (2019) Figure 5
#prasath_data = pd.read_csv('prasath_data.csv')

# Compute asymptotic data using (4.8) with sign corrected singular term
stokes_num = rad_p ** 2 / (3 * t_scale * nu_f)
sigma = particle1.p.sigma
beta  = np.zeros(len(rho_p))
r 	  = np.zeros(len(rho_p))
alpha = np.zeros(len(rho_p))
gamma = np.zeros(len(rho_p))

for i in range(len(rho_p)):
	beta[i] = rho_p[i] / rho_f
	r[i] = (1 + 2 * beta[i]) / 3
	alpha[i] = 1 / (r[i] * stokes_num)
	gamma[i] = 1 / r[i] * np.sqrt(3 / stokes_num)	

#c = np.pi * sigma / alpha
#q0_1_asymp = c[0] - sigma * gamma[0] / (rad_p ** 2 * np.sqrt(np.pi * taxis[1:]))
#q0_2_asymp = c[1] - sigma * gamma[1] / (rad_p ** 2 * np.sqrt(np.pi * taxis[1:]))
#q0_3_asymp = c[2] - sigma * gamma[2] / (rad_p ** 2 * np.sqrt(np.pi * taxis[1:]))
c = -(np.pi / alpha * (r - 1))
q0_1_asymp = c[0] - (1 - r[0]) * gamma[0] \
				  / (alpha[0] ** 2 * np.sqrt(np.pi * taxis[1:]))
q0_2_asymp = c[1] - (1 - r[1]) * gamma[1] \
				  / (alpha[1] ** 2 * np.sqrt(np.pi * taxis[1:]))
q0_3_asymp = c[2] - (1 - r[2]) * gamma[2] \
				  / (alpha[2] ** 2 * np.sqrt(np.pi * taxis[1:]))

# Generate plots
title_fs, label_fs, tick_fs = 16, 14, 12 

plt.figure(figsize=(16, 9))
plt.suptitle(r'Figure 5 Recreation with $ \sigma = $ {:.2f} and '.format(sigma)
			 + r'$ \vec{{q}}_0 = $ ({:.0f}, {:.0f})'.format(u0, v0),
			 fontsize=title_fs)
# Velocity plot
plt.subplot(121)
plt.xlabel(r'$t$', fontsize=label_fs)
plt.ylabel('$ 1 + q^{(2)}(0, t) $', fontsize=label_fs)
plt.axis([0, 10, 0.95, 1.02])
plt.xticks([0, 5, 10])
plt.yticks([0.96, 0.98, 1.00, 1.02])
plt.tick_params(labelsize=tick_fs)
plt.plot(taxis, 1 + q0_1_vec[:, 1], c='mediumvioletred',
			label=r'my numerics, $\beta = 5$')
plt.plot(taxis, 1 + q0_2_vec[:, 1], c='hotpink',
			label=r'my numerics, $\beta = 1$')
plt.plot(taxis, 1 + q0_3_vec[:, 1], c='orange', lw=1.5,
		 label=r'my numerics, $\beta = 0.01$')
plt.plot(taxis[1:], 1 + q0_1_asymp, c='mediumvioletred', ls=':',
		 label=r'eq (4.8), $\beta = 5$')
plt.plot(taxis[1:], 1 + q0_2_asymp, c='hotpink', ls=':',
		 label=r'eq (4.8), $\beta = 1$')
plt.plot(taxis[1:], 1 + q0_3_asymp, c='orange', ls=':',
		 label=r'eq (4.8), $\beta = 0.01$')
#plt.plot('t_heavy', 'q_heavy', '.-b', data=prasath_data, 
#		 label=r'Prasath data, $\beta = 5$')
#plt.plot('t_med', 'q_med', '.-g', data=prasath_data,
#		 label=r'Prasath data, $\beta = 1$')
#plt.plot('t_light', 'q_light', '.-r', data=prasath_data,
#		 label=r'Prasath data, $\beta = 0.01$')
#plt.plot('asymp_t_heavy', 'asymp_q_heavy', 'o:b', mfc='none', mec='b',
#		 data=prasath_data, label=r'Prasath asymptotic data, $\beta = 5$')
#plt.plot('asymp_t_med', 'asymp_q_med', 'o:g', mfc='none', mec='g',
#		 data=prasath_data, label=r'Prasath asymptotic data, $\beta = 1$')
#plt.plot('asymp_t_light', 'asymp_q_light', 'o:r', mfc='none', mec='r',
#		 data=prasath_data, label=r'Prasath asymptotic data, $\beta = 0.01$')
plt.legend()

# Trajectory plot
plt.subplot(122)
plt.xlabel('$y^{(1)}(t)$', fontsize=label_fs)
plt.ylabel('$y^{(2)}(t)$', fontsize=label_fs)
plt.axis([0, 0.04, 0, 0.035])
plt.xticks([0, 0.01, 0.02, 0.03, 0.04])
plt.yticks([0, 0.01, 0.02, 0.03])
plt.tick_params(labelsize=tick_fs)
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 2))
plt.subplot(122).xaxis.set_major_formatter(formatter)
plt.subplot(122).yaxis.set_major_formatter(formatter)
plt.plot(pos1_vec[:, 0], pos1_vec[:, 1], c='mediumvioletred',
		 label=r'my numerics, $\beta = 5$')
plt.plot(pos1_vec[:, 0], pos2_vec[:, 1], c='hotpink',
		 label=r'my numerics, $\beta = 1$')
plt.plot(pos1_vec[:, 0], pos3_vec[:, 1], c='orange',
		 label=r'my numerics, $\beta = 0.01$')
#plt.plot('y1_heavy', 'y2_heavy', '.-b', data=prasath_data,
#		 label=r'Prasath data, $\beta = 5$')
#plt.plot('y1_med', 'y2_med', '.-g', data=prasath_data,
#		 label=r'Prasath data, $\beta = 1$')
#plt.plot('y1_light', 'y2_light', '.-r', data=prasath_data,
#		 label=r'Prasath data, $\beta = 0.01$')
#plt.plot('asymp_y1_heavy', 'asymp_y2_heavy', 'o:b', mfc='none', mec='b',
#		 data=prasath_data, label=r'Prasath data, $\beta = 5$')
#plt.plot('asymp_y1_med', 'asymp_y2_med', 'o:g', mfc='none', mec='g',
#		 data=prasath_data, label=r'Prasath data, $\beta = 1$')
#plt.plot('asymp_y1_light', 'asymp_y2_light', 'o:r', mfc='none', mec='r',
#		 data=prasath_data, label=r'Prasath data, $\beta = 0.01$')
plt.legend()

plt.tight_layout()
plt.savefig(save_plot_to + 'c01_FIGURE_QUIESCENT.pdf', format='pdf', dpi=600)
plt.show()
print("\007")

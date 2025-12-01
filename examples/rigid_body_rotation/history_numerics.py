import warnings
import numpy as np
import pandas as pd

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import rotating_flow as fl
from models import my_system as ts
from examples.rigid_body_rotation.numerics import STOKES_HAT, X_0, Z_0
from examples.rigid_body_rotation.numerics import R as R_DAITCHE

R = 2 * R_DAITCHE / 3
INCLUDE_HISTORY = True
INCLUDE_H = True
KEYS = ['t', 'H_x', 'H_z', 'history_x', 'history_z', 'delta_t']
OUT_FILE = '../data/rigid_body_rotation/history.csv'

def main():
	"""
	Record the numerical solutions for the history force on a rotating particle.

	The conditions of the simulations were chosen based on the rigid body
	rotation example in [1]. Solutions are saved to the
	`data/rigid_body_rotation` directory.
	
	References
	----------
	[^1]: [A. Daitche (2013).](https://doi.org/10.1016/j.jcp.2013.07.024)
		  Advection of inertial particles in the presence of the history force:
		  Higher order numerical schemes. *Journal of Computational Physics*
		  254, 93–106.
	"""
	# initialize variables for the transport system
	particle = prt.Particle(STOKES_HAT)
	flow = fl.RotatingFlow()
	system = ts.MyTransportSystem(particle, flow, R)

	# initialize variables for the numerical simulations
	t_final, delta_t = 10, 1e-2
	xdot_0, zdot_0 = flow.velocity(X_0, Z_0)
	y = [X_0, Z_0, xdot_0, zdot_0]
	t = np.arange(0, t_final, delta_t)
	results = {key: [] for key in KEYS}

	# run simulation and store results
	warnings.filterwarnings('ignore')
	_, _, _, _, t, _, _, _, _, _, _, _, _, h_x, h_z, history_x, \
	   history_z = system.maxey_riley(t, y, INCLUDE_HISTORY,
									  include_h=INCLUDE_H)
	results = update_results(results, [t, h_x, h_z, history_x, history_z],
							 [delta_t])
	# run simulations for various delta_t's and store only the values at t = 0
	t_final = 0.05
	for delta_t in [5e-3, 1e-3, 5e-4]:
		t = np.arange(0, t_final, delta_t)
		_, _, _, _, t, _, _, _, _, _, _, _, _, h_x, h_z, history_x, \
		   history_z = system.maxey_riley(t, y, INCLUDE_HISTORY,
										  include_h=INCLUDE_H)
		results = update_results(results, [], [t[0], h_x[0], h_z[0],
								 history_x[0], history_z[0], delta_t])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

if __name__ == '__main__':
	main()

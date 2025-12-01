import numpy as np
import pandas as pd

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import rotating_flow as fl
from models import rotating_system as ts

SCALE = 2 / 3				# scale used for parameter translation
R = 0.75					# density ratio
STOKES_HAT = SCALE * 0.3
X_0, Z_0 = 1, 0				# initial particle position
T_FINAL = 100				# total time
DELTA_T = 1e-2				# timestep
OUT_FILE = '../data/rigid_body_rotation/numerics.csv'

def main():
	"""
	Run numerical simultions for a rotating rigid body.

	The simulations reproduce the numerical results from [1] Figures 3 and 4.
	Results are saved to the `data/rigid_body_rotation` directory.
	
	References
	----------
	[^1]: [A. Daitche (2013).](https://doi.org/10.1016/j.jcp.2013.07.024)
		  Advection of inertial particles in the presence of the history force:
		  Higher order numerical schemes. *Journal of Computational Physics*
		  254, 93–106.
	"""
	# create the Particle, Flow, and TransportSystem objects
	particle = prt.Particle(STOKES_HAT)
	flow = fl.RotatingFlow()
	system = ts.RotatingTransportSystem(particle, flow, R)

	# create dict to store sols, initialize variables for the simulations
	results = {'t': [], 'x': [], 'z': [], 'order': []}
	xdot_0, zdot_0 = flow.velocity(X_0, Z_0)
	y = [X_0, Z_0, xdot_0, zdot_0]
	t = np.arange(0, T_FINAL, DELTA_T)

	# run simulations and store solutions
	for order in [1, 2, 3]:
		x, z, _, _, _ = system.maxey_riley(t, y, order)
		results = update_results(results, [t, x, z], [order])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file
		
if __name__ == '__main__':
	main()

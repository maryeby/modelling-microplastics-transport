import numpy as np
import pandas as pd

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import rotating_flow as fl
from models import my_system as ts

SCALE = 2 / 3				# scale used for parameter translation
R = 0.75					# density ratio
STOKES_HAT = SCALE * 0.3
X_0, Z_0 = 1, 0				# initial particle position
DELTA_T = 5e-4				# timestep
T_FINAL = 1					# total time
OUT_FILE = '../data/rigid_body_rotation/mini_numerics.csv'

def main():
	r"""Run numerical simultions for a rotating rigid body at small time $t$."""
	# create the Particle, Flow, and TransportSystem objects
	particle = prt.Particle(STOKES_HAT)
	flow = fl.RotatingFlow()
	system = ts.MyTransportSystem(particle, flow, R)

	# create dict to store sols, initialize variables for the simulations
	results = {'t': [], 'history_x': [], 'history_z': []}
	xdot_0, zdot_0 = flow.velocity(X_0, Z_0)
	y = [X_0, Z_0, xdot_0, zdot_0]
	t = np.arange(0, T_FINAL, DELTA_T)

	# run simulations and store solutions
	history_x, history_z = system.maxey_riley(t, y, True)[-2:]
	results = update_results(results, [t, history_x, history_z], [])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file
		
if __name__ == '__main__':
	main()

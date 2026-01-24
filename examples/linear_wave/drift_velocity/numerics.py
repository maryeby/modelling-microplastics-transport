import warnings
import numpy as np
import pandas as pd

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import linear_wave as fl
from models import my_system as ts

# wave conditions
DEPTH = 10
AMPLITUDE = 0.02
WAVELENGTH = 1

# simulation conditions
SCALE = 2 / 3
BETA = 0.9
R = SCALE * BETA
NUM_PERIODS = 30
DELTA_T = 5e-3
KEYS = ['t', 'x', 'z', 'u_bar', 'w_bar', 'history']

# particle conditions
STOKES_NUM = R * 0.157 * 0.125
X_0, Z_0 = 0, 0

OUT_FILE = '../../data/linear_wave/drift_vel_numerics.csv'

def main():
	"""
	Run simulations and compute the drift velocity of a particle in a wave.

	The simulations produce numerical solutions for the transport of a
	negatively buoyant particle in a linear wave of deep water, with and without
	history effects. The period-averaged Stokes drift velocity is also computed.
	Results are saved to the `data/linear_wave` directory.

	See Also
	--------
	models.my_system.compute_alternate_drift_velocity()
	"""
	# create the Particle, Flow, and TransportSystem objects
	particle = prt.Particle(STOKES_NUM)
	wave = fl.LinearWave(DEPTH, AMPLITUDE, WAVELENGTH)
	system = ts.MyTransportSystem(particle, wave, R)

	# create dict to store sols, initialize variables for the simulations
	results = {key: [] for key in KEYS}
	xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
	y = [X_0, Z_0, xdot_0, zdot_0]
	warnings.filterwarnings('ignore')

	for history in [False, True]:
		# run simulation and store results
		t = np.arange(0, wave.period * NUM_PERIODS, DELTA_T)
		x, z, xdot, zdot, t = system.maxey_riley(t, y, history)[:5]
		results = update_results(results, [t, x, z], [None, None, history])

		# compute drift velocity and store results
		x, z, u, w, t = ts.compute_alternate_drift_velocity(x, z, xdot, zdot, t,
															NUM_PERIODS)
		results = update_results(results, [t, x, z, u, w],
								[history])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

if __name__ == '__main__':
	main()

import warnings
import numpy as np
import pandas as pd
from tqdm.contrib.itertools import product

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import water_wave as fl
from models import my_system as ts

# particle conditions
STOKES_NUMS = [0.01, 0.1, 0.3]
X_0, Z_0 = 0, 0

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
HIDE_PROGRESS = True
OUT_FILE = '../../../data/water_wave/st_numerics.csv'

def main():
	"""
	Run numerical simulations for negatively buoyant particles in a wave.

	Simulations are run with and without history effects, with various Stokes
	numbers, and in linear waves of deep water. The period-averaged Stokes
	drift velocity is also computed. Results are saved to the `data/water_wave`
	directory.

	See Also
	--------
	models.my_system.compute_drift_velocity
	"""
	# create dict to store sols, initialize variables for the simulations
	keys = ['t', 'x', 'z', 'u_bar', 'w_bar', 'St', 'history']
	results = {key: [] for key in keys}
	wave = fl.WaterWave(DEPTH, AMPLITUDE, WAVELENGTH)
	xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
	y = [X_0, Z_0, xdot_0, zdot_0]
	t = np.arange(0, NUM_PERIODS, DELTA_T)

	warnings.filterwarnings('ignore')
	for stokes_num, history in product(STOKES_NUMS, [False, True]):
		# create Particle and TransportSystem objects
		particle = prt.Particle(stokes_num)
		system = ts.MyTransportSystem(particle, wave, R)
		t = np.arange(0, NUM_PERIODS, DELTA_T)

		# run numerical simulation and store solutions
		x, z, xdot, _, t, _, _, _, _, _, _, _, _, _, \
		   _ = system.maxey_riley(t, y, history, HIDE_PROGRESS)
		results = update_results(results, [t, x, z], 
								[None, None, stokes_num, history])

		# compute drift velocity and store solutions
		x, z, u, w, t = ts.compute_drift_velocity(x, z, xdot, t)
		results = update_results(results, [t[1:], x[1:], z[1:], u, w],
								[stokes_num, history])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

if __name__ == '__main__':
	main()

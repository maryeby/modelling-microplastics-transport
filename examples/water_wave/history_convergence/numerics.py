import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import water_wave as fl
from models import my_system as ts

# particle conditions
STOKES_NUM = 0.01
X_0, Z_0 = 0, 0

# wave conditions
DEPTH = 10
AMPLITUDE = 0.02
WAVELENGTH = 1

# simulation conditions
SCALE = 2 / 3
BETA = 0.9
R = BETA * SCALE
DELTA_TS = np.linspace(5e-3, 3e-4, 10)
INCLUDE_HISTORY = True
HIDE_PROGRESS = True
OUT_FILE = '../../data/water_wave/history_convergence.csv'

def main():
	"""
	Compute numerical solutions for the history force at time *t* = 0.

	The value of the history force is recorded from simulations run with various
	time step sizes. Results are saved to the `data/water_wave` directory.
	"""
	# create Particle, Wave, and TransportSystem objects
	particle = prt.Particle(STOKES_NUM)
	wave = fl.WaterWave(DEPTH, AMPLITUDE, WAVELENGTH)
	system = ts.MyTransportSystem(particle, wave, R)

	results = {'initial_history_x': [], 'initial_history_z': [], 'delta_t': []}
	xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
	y = [X_0, Z_0, xdot_0, zdot_0]
	warnings.filterwarnings('ignore')

	for delta_t in tqdm(DELTA_TS):
		# initialize time series and run simulation
		t_final = delta_t * 8
		t = np.arange(0, t_final, delta_t)
		_, _, _, _, _, _, _, _, _, _, _, _, _, \
		   history_x, history_z = system.maxey_riley(t, y, INCLUDE_HISTORY,
													 HIDE_PROGRESS)
		# store only initial values of history
		results = update_results(results, [], [history_x[0], history_z[0],
											   delta_t])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

if __name__ == '__main__':
	main()

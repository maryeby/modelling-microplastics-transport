import warnings
import numpy as np
import pandas as pd

from utils.data_tools import update_results
from transport_framework import particle as prt 
from models import water_wave as fl
from models import my_system as ts

# wave conditions
DEPTH = 100
AMPLITUDE = 0.02
WAVELENGTH = 1.5

# particle conditions
STOKES_NUM = 0.1
X_0, Z_0 = 0, 0

# simulation conditions
SCALE = 2 / 3
BETA = 0.99
R = SCALE * BETA
NUM_PERIODS = 1000
DELTA_TS = [2.5e-3, 5e-3, 1e-2]
INCLUDE_HISTORY = False
OUT_FILE = '../data/water_wave/sensitivity.csv'

def main():
	"""Run simulations with varying timesteps to test the model sensitivity."""
	# create the Particle, Flow, and TransportSystem objects
	particle = prt.Particle(STOKES_NUM)
	wave = fl.WaterWave(DEPTH, AMPLITUDE, WAVELENGTH)
	system = ts.MyTransportSystem(particle, wave, R)

	# create dict to store results and set initial particle velocity
	results = {'delta_x': [], 'delta_t': []}
	xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
	y = [X_0, Z_0, xdot_0, zdot_0]
	warnings.filterwarnings('ignore')

	# run a simulation for each timestep and store the horizontal displacement
	for delta_t in DELTA_TS:
		t = np.arange(0, NUM_PERIODS * wave.period, delta_t)
		x = system.maxey_riley(t, y, INCLUDE_HISTORY)[0]
		delta_x = np.abs(x[-1]) - np.abs(X_0)
		results = update_results(results, [], [delta_x, delta_t])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)

if __name__ == '__main__': main()

import warnings
import numpy as np
import pandas as pd
from parallelbar import progress_map

from utils.data_tools import update_results
from transport_framework import particle as prt 
from models import linear_wave as fl
from models import my_system as ts

# wave conditions
DEPTH = 100
AMPLITUDE = 0.02
WAVELENGTH = 1.5

# particle conditions
STOKES_HAT = 0.15
X_0, Z_0 = 0, 0

# simulation conditions
R = 0.66
NUM_PERIODS = 100
NUM_CPUS = None
TIMEOUT = None
DELTA_TS = [2.5e-3, 5e-3, 1e-2]
INCLUDE_HISTORY = False
HIDE_PROGRESS = True
OUT_FILE = '../../data/linear_wave/sensitivity.csv'

def main():
	"""Run simulations with varying timesteps to test the model sensitivity."""
	results = {'delta_x': [], 'delta_t': []}
	warnings.filterwarnings('ignore')
	sols = progress_map(run_numerics, DELTA_TS, process_timeout=TIMEOUT,
						n_cpu=NUM_CPUS, total=len(DELTA_TS))
	for sol in sols: results = update_results(results, [], sol)
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)

def run_numerics(delta_t):
	"""Run numerical simulations for various time step sizes."""
	print(f'Started {delta_t:.1e} simulation')
	particle = prt.Particle(STOKES_HAT)
	wave = fl.LinearWave(DEPTH, AMPLITUDE, WAVELENGTH)
	system = ts.MyTransportSystem(particle, wave, R)
	xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
	y = [X_0, Z_0, xdot_0, zdot_0]
	t = np.arange(0, NUM_PERIODS * wave.period, delta_t)
	x = system.maxey_riley(t, y, INCLUDE_HISTORY, HIDE_PROGRESS)[0]
	delta_x = np.abs(x[-1]) - np.abs(X_0)
	return [delta_x, delta_t]

if __name__ == '__main__': main()

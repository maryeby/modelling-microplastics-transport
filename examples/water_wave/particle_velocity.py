import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import hilbert, find_peaks

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import water_wave as fl
from models import my_system as ts

# wave conditions
DEPTH = 10
AMPLITUDE = 0.02
WAVELENGTH = 1.5

# particle conditions
SCALE = 2 / 3		# for parameter translations
STOKES_NUM = SCALE
X_0, Z_0 = 0, 0

# simulation conditions
BETA = 0.99			# density ratio
R = SCALE * BETA	# density ratio
NUM_PERIODS = 30
DELTA_T = 5e-3		# timestep size
INCLUDE_HISTORY = True
KEYS = ['t', 'curve', 'curve_type']
OUT_FILE = '../data/water_wave/particle_velocity_numerics.csv'

def main():
	"""Run a simulation and fit a curve to the particle velocity decay."""
	# create the Particle, Flow, and TransportSystem objects
	particle = prt.Particle(STOKES_NUM)
	wave = fl.WaterWave(DEPTH, AMPLITUDE, WAVELENGTH)
	system = ts.MyTransportSystem(particle, wave, R)
	results = {key: [] for key in KEYS}

	# initialize time series and set initial particle velocity = fluid velocity
	t = np.arange(0, NUM_PERIODS * wave.period, DELTA_T)
	xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
	y = [X_0, Z_0, xdot_0, zdot_0]  # initial particle position and velocity
	warnings.filterwarnings('ignore')

	# run simulation
	_, _, xdot, _, t, _, _, _, _, _, _, _, _, _, \
	   _ = system.maxey_riley(t, y, INCLUDE_HISTORY)
	results = update_results(results, [t, xdot], ['xdot'])

	# compute and store Hilbert transform
	analytical_curve = hilbert(xdot)
	hilbert_transform = np.abs(analytical_curve)
	trun = find_peaks(hilbert_transform)[0][0]
	cate = -40
	results = update_results(results, [t[trun:cate],
							 hilbert_transform[trun:cate]], ['envelope'])

	# fit curve to envelope
	coefficients, cov = curve_fit(f, t[trun:cate], hilbert_transform[trun:cate])
	a, decay_rate, offset = coefficients
	curve = f(t[trun:cate], a, decay_rate, offset)
	results = update_results(results, [t[trun:cate], curve], ['decay'])
	print(f'HT decay rate: {decay_rate:.2f}')
	
	# fit a curve to the peaks of the particle velocity
	indices = find_peaks(xdot)[0]
	peaks = xdot[indices]
	t_peaks = t[indices]
	coefficients, cov = curve_fit(f, t_peaks, peaks)
	a, decay_rate, offset = coefficients
	curve = f(t, a, decay_rate, offset)
	results = update_results(results, [t, curve], ['fitted'])
	results['t_peaks'] = t_peaks.tolist()
	results['peaks'] = peaks.tolist()
	print(f'CF decay rate: {decay_rate:.2f}')

	# store results
	df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value \
					  in results.items()]))
	df.to_csv(OUT_FILE, index=False)
	
def f(x, a, b, offset): return a * np.exp(-x * b) + offset

if __name__ == '__main__': main()

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
SCALE = 2 / 3						# for parameter translations
STOKES_NUMS = [0.10, 0.10, 1, 1]
X_0, Z_0 = 0, 0

# simulation conditions
BETAS = [0.81, 0.81, 0.95, 0.95]	# density ratio
NUM_PERIODS = 5
DELTA_T = 5e-3						# timestep size
INCLUDE_HISTORY = [True, False, True, False]
KEYS = ['t', 'curve', 'curve_type', 'St', 'beta', 'history']
OUT_FILE = '../data/water_wave/particle_velocity_numerics.csv'

def main():
	"""Run simulations and fit curves to the particle velocity decay."""
	results = {key: [] for key in KEYS}
	t_peaks_list, peaks_list = [], []
	warnings.filterwarnings('ignore')
	for stokes_num, beta, history in zip(STOKES_NUMS, BETAS, INCLUDE_HISTORY):
		# create the Particle, Flow, and TransportSystem objects
		particle = prt.Particle(stokes_num)
		wave = fl.WaterWave(DEPTH, AMPLITUDE, WAVELENGTH)
		system = ts.MyTransportSystem(particle, wave, beta * SCALE)

		# initialize time series and set initial particle velocity
		t = np.arange(0, NUM_PERIODS * wave.period, DELTA_T)
		xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
		y = [X_0, Z_0, xdot_0, zdot_0]  # initial particle position and velocity

		# run simulation
		_, _, xdot, _, t, _, _, _, _, _, _, _, _, _, \
		   _ = system.maxey_riley(t, y, history)
		results = update_results(results, [t, xdot], ['xdot', stokes_num, beta,
													  history])
		# fit a curve to the peaks of the particle velocity
		indices = find_peaks(xdot)[0]
		peaks = xdot[indices]
		t_peaks = t[indices]
		coefficients, cov = curve_fit(f, t_peaks, peaks)
		a, decay_rate, offset = coefficients
		curve = f(t, a, decay_rate, offset)
		results = update_results(results, [t, curve], ['fitted', stokes_num, 
													   beta, history])
		t_peaks_list += t_peaks.tolist()
		peaks_list += peaks.tolist()
		print(f'decay rate: {decay_rate:.2f}\n')
	results['t_peaks'] = t_peaks.tolist()
	results['peaks'] = peaks.tolist()

	# store results
	df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value \
					  in results.items()]))
	df.to_csv(OUT_FILE, index=False)
	
def f(x, a, b, offset): return a * np.exp(-x * b) + offset

if __name__ == '__main__': main()

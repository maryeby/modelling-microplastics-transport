import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import hilbert, find_peaks

from utils.data_tools import update_results, print_characteristic_params
from transport_framework import particle as prt
from models import linear_wave as fl
from models import my_system as ts

# wave conditions
DEPTH = 10
AMPLITUDE = 0.02
WAVELENGTH = 1.5

# particle conditions
STOKES_HATS = np.round([73 / 36, 73 / 36, 335 / 33, 335 / 33], 5)
X_0, Z_0 = 0, 0

# simulation conditions
RS = [0.54, 0.54, 0.66, 0.66]		# density ratio
NUM_PERIODS = 5
DELTA_T = 5e-3						# timestep size
INCLUDE_HISTORY = [True, False, True, False]
KEYS = ['t', 'curve', 'curve_type', 'St', 'Sthat', 'R', 'history']
OUT_FILE = '../../data/linear_wave/particle_velocity_numerics.csv'

def main():
	r"""
	Simulate particle transport and fit curves to its velocity decay.

	To fit the curves and determine the rate of decay *b*, the expression
	$$f(x, a, b) = ae^{-bx}$$
	is employed.
	"""
	results = {key: [] for key in KEYS}
	t_peaks_list, peaks_list = [], []
	warnings.filterwarnings('ignore')
	for stokes_hat, r, history in zip(STOKES_HATS, RS, INCLUDE_HISTORY):
		# create the Particle, Flow, and TransportSystem objects
		particle = prt.Particle(stokes_hat)
		wave = fl.LinearWave(DEPTH, AMPLITUDE, WAVELENGTH)
		system = ts.MyTransportSystem(particle, wave, r)

		# initialize time series and set initial particle velocity
		t = np.arange(0, NUM_PERIODS * wave.period, DELTA_T)
		xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
		y = [X_0, Z_0, xdot_0, zdot_0]  # initial particle position and velocity

		# run simulation
		xdot, _, t = system.maxey_riley(t, y, history)[2:5]
		results = update_results(results, [t, xdot], ['xdot', system.stokes_num,
													  stokes_hat, r, history])
		# fit a curve to the peaks of the particle velocity
		indices = find_peaks(xdot)[0]
		indices = np.insert(indices, 0, 0)
		peaks = xdot[indices]
		t_peaks = t[indices]
		coefficients, cov = curve_fit(f, t_peaks, peaks)
		a, decay_rate = coefficients
		curve = f(t, a, decay_rate)
		results = update_results(results, [t, curve], ['fitted',
								 system.stokes_num, stokes_hat, r, history])
		t_peaks_list += t_peaks.tolist()
		peaks_list += peaks.tolist()
		if history:
			print(f'decay rate: {decay_rate:.2f}')
			print_characteristic_params(particle, wave, system)
		else:
			print(f'decay rate: {decay_rate:.2f}\n')
	results['t_peaks'] = t_peaks.tolist()
	results['peaks'] = peaks.tolist()

	# store results
	df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value \
					  in results.items()]))
	df.to_csv(OUT_FILE, index=False)
	
def f(x, a, b): return a * np.exp(-x * b)

if __name__ == '__main__': main()

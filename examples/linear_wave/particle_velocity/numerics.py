import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress

from utils.data_tools import update_results, print_characteristic_params, \
							 print_parameter, f, g
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
RS = [0.54, 0.54, 0.66, 0.66]	# density ratios
NUM_PERIODS = 5
DELTA_T = 5e-3					# timestep size
INCLUDE_HISTORY = [True, False, True, False]
KEYS = ['t', 'xdot', 'fitted_curve', 'decay_curve', 'R^2', 'decay_rate',
		'Sthat', 'S', 'R', 'history']
OUT_FILE = '../../data/linear_wave/particle_velocity_numerics.csv'

def main():
	r"""
	Simulate particle transport and fit curves to its velocity decay.

	To fit the curves and determine the rate of decay $\delta$, the expression
	$$f(t, A, \delta, \phi, \text{offset}) = Ae^{-\delta t}\sin(t + \phi)
	+ \text{offset}$$ is employed. To create the decay curve to be plotted, the
	equation $$g(t, A, \delta) = Ae^{-\delta t}$$ is used, evaluated with the
	$A$, $\delta$ provided from the curve fitting.

	See Also
	--------
	utils.data_tools.f()
	utils.data_tools.g()
	"""
	results = {key: [] for key in KEYS}
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

		# run simulation, fit a curve to the data and compute R^2 value
		xdot, _, t = system.maxey_riley(t, y, history)[2:5]
		coefficients, _ = curve_fit(f, t, xdot)
		a, decay_rate, phi, offset = coefficients
		fitted_curve = f(t, a, decay_rate, phi, offset)
		decay_curve = g(t, a, -decay_rate, 0)
		stats = linregress(xdot, fitted_curve)
		rsq = stats.rvalue * stats.rvalue

		# print parameters and update results
		if history: print_characteristic_params(wave, particle, system)
		print(' ', history)
		print_parameter('decay_rate', decay_rate)
		print_parameter('R^2', rsq)
		print()
		results = update_results(results, [t, xdot, fitted_curve, decay_curve],
								[rsq, decay_rate, stokes_hat,
								 stokes_hat / system.gamma, r, history])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)
	
if __name__ == '__main__': main()

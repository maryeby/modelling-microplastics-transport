import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress
from itertools import product
from tqdm import tqdm

from utils.data_tools import update_results, extract_data, print_parameter, f
from utils.plot import initialize_figure as fig
from transport_framework import particle as prt
from models import stokes_wave as fl
from models import my_system as ts
from examples.linear_wave.particle_velocity.numerics import DEPTH, WAVELENGTH, \
	 NUM_PERIODS, DELTA_T, X_0, Z_0
from examples.stokes_wave.particle_velocity.numerics import STOKES_HATS, RS, \
	 AMPLITUDES, HISTORY_A, NUM_TASKS, INCLUDE_HISTORY
from examples.stokes_wave.particle_velocity.numerics import OUT_FILE as IN_FILE

KEYS = ['curve', 'decay_rate', 'R^2', 'A\'', 'epsilon', 'S', 'Sthat', 'R',
		'history']
TOL = 0.98
EPSILON_TO_SHOW = 0.03979
OUT_FILE = '../../data/stokes_wave/particle_velocity_analysis.csv'

def main():
	r"""
	Fit curves to the horizontal particle velocity decay.

	To fit the curves and determine the rate of decay $\delta$, the expression
	$$f(t, A, \delta, \phi, \text{offset}) = Ae^{-\delta t}\sin(t + \phi)
	+ \text{offset}$$ is employed.

	See Also
	--------
	utils.data_tools.f()
	"""
	numerics = pd.read_csv(IN_FILE)
	warnings.filterwarnings('ignore')
	results = {key: [] for key in KEYS}
	sthats, amplitudes = zip(*product(STOKES_HATS, AMPLITUDES))
	rs, history = zip(*product(RS, INCLUDE_HISTORY))
	names = ['xdot', 't', 'S', 'epsilon']
	for sthat, r, a, h in tqdm(zip(sthats, rs, amplitudes, history),
							   total=NUM_TASKS):
		show_plot = sthat == STOKES_HATS[1] and r == RS[1] \
					and epsilon == EPSILON_TO_SHOW and h
		params = {'Sthat': sthat, 'R': r, 'A\'': a, 'history': h}
		xdot, t, s, epsilon = extract_data(names, numerics, params)
		xdot, t = xdot.to_numpy(), t.to_numpy()
		s, epsilon = s.iloc[0], epsilon.iloc[0]
		if show_plot:
			print_parameter('S', s)
			print_parameter('R', r)
			print_parameter('epsilon', epsilon)
		sols = compute_decay(xdot, t, show_plot)
		sols += [a, epsilon, s, sthat, r, h]
		results = update_results(results, [sols[0]], sols[1:])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)
	plt.show()

def compute_decay(xdot, t, plot_results=False):
	"""Return the curve fit & decay rate of the horizontal particle velocity."""
	coefficients, _ = curve_fit(f, t, xdot)#, p0=initial_guess)
	a, decay_rate, phi, offset = coefficients
	curve = f(t, a, decay_rate, phi, offset)
	stats = linregress(xdot, curve)
	rsq = stats.rvalue * stats.rvalue
	if plot_results:
		fig(r'$t$', r'$\dot{x}$')
		plt.plot(t, xdot, '-k')
		plt.plot(t, curve, '--k')
	return [curve, decay_rate, rsq]

if __name__ == '__main__': main()

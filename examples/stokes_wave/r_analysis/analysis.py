import warnings
import numpy as np
import pandas as pd
import scipy.constants as constants
from scipy.optimize import curve_fit
from tqdm.contrib.itertools import product

from utils.data_tools import extract_data, update_results, g
from models import stokes_wave as fl
from examples.linear_wave.r_analysis.numerics import DEPTH, WAVELENGTH, RS
from examples.stokes_wave.r_analysis.numerics import AMPLITUDES
from examples.stokes_wave.r_analysis.numerics import OUT_FILE as IN_FILE

A = AMPLITUDES[1]
EXT_RANGE = 300
MAXFEV = 100000
KEYS = ['z', 'u', 'R', 'history', 'analytical']
OUT_FILE = '../../data/stokes_wave/r_analysis.csv'

def main():
	r"""
	Compute the drift velocity of particles in a wave, fit a curve to the data.

	The average horizontal Stokes drift velocity is numerically computed for
	particles of varying buoyancies in linear waves of deep water, and curves
	are fit to the resulting data points. For neutrally buoyant particles, one
	data point is produced for each simulation; the drift velocity is averaged
	over the wave periods, then averaged over the trajectory.

	Since there is an analytical solution[^1] for the Stokes drift velocity of
	neutrally buoyant particles, the analytical solutions are computed for all
	neutrally buoyant simulations, rather than fitting a curve to the numerical
	solutions. For other particles, the drift velocity is averaged over each
	wave period, and single curve is fit to the numerical solutions of each
	simulation. The function used to fit the curve to the data is,
	$$g(x, A, \delta, \text{offset}) = Ae^{\delta x} + \text{offset}.$$ Results
	are saved to the `data/stokes_wave` directory.

	See Also
	--------
	models.my_system.compute_drift_velocity()
	utils.data_tools.g()

	References
	----------
	[^1]: [T. S. van den Bremer & Ø. Breivik (2018)](
		  https://doi.org/10.1098/rsta.2017.0104) Stokes drift.
		  *Philosophical Transactions of the Royal Society A: Mathematical,
		  Physical and Engineering Sciences* 376(2111), 20170104.
	"""
	# read data and create a dict to store results
	numerics = pd.read_csv(IN_FILE)
	results = {key: [] for key in KEYS}

	# initialize wave conditions
	wave = fl.StokesWave(DEPTH, A, WAVELENGTH)
	k = wave.wavenum
	analytical = False

	# analysis for non-neutrally buoyant particles
	warnings.filterwarnings('ignore')
	for r, history in product(RS[1:], [True, False]):
		# extract and normalize data
		names = ['z_bar', 'u_d_bar', 'mean_speed']
		params = {'R': r, 'history': history}
		z_bar, u_d_bar, u_bar = extract_data(names, numerics, params)
		z_bar = z_bar.to_numpy()
		u_d_bar = u_d_bar.to_numpy()
		extended_range = np.linspace(0, z_bar[-1], EXT_RANGE) if r < 2 / 3 else\
						 np.linspace(-7, 0, EXT_RANGE)

		# fit a curve to the numerical data
		coefficients, _ = curve_fit(g, z_bar, u_d_bar, maxfev=MAXFEV)
		a, delta, offset = coefficients
		u_d_bar = g(extended_range, a, delta, offset)
		results = update_results(results, [extended_range, u_d_bar],
								[r, history, analytical])

	# compute and store analytical solutions
	r, history, analytical = RS[0], None, True
	h = k * DEPTH
	z = np.linspace(0, -7, EXT_RANGE)
	u_d = np.cosh(2 * (z + h)) / (2 * np.sinh(h) ** 2)
	results = update_results(results, [z, u_d], [r, history, analytical])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

if __name__ == '__main__':
	main()

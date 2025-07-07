import warnings
import numpy as np
import pandas as pd
import scipy.constants as constants
from scipy.optimize import curve_fit
from tqdm.contrib.itertools import product

from utils.data_tools import extract_data, update_results
from models import water_wave as fl
from examples.water_wave.beta_analysis.numerics import AMPLITUDE as A
from examples.water_wave.beta_analysis.numerics import DEPTH, WAVELENGTH, BETAS
from examples.water_wave.beta_analysis.numerics import OUT_FILE as IN_FILE

EXT_RANGE = 300
MAXFEV = 100000
OUT_FILE = '../../data/water_wave/beta_analysis.csv'

def main():
	"""
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
	simulation. Results are saved to the `data/water_wave` directory.

	See Also
	--------
	models.my_system.compute_drift_velocity

	References
	----------
	[^1]: [T. S. van den Bremer & Ø. Breivik (2018)](
		  https://doi.org/10.1098/rsta.2017.0104) Stokes drift.
		  *Philosophical Transactions of the Royal Society A: Mathematical,
		  Physical and Engineering Sciences* 376(2111), 20170104.
	"""
	# read data and create a dict to store results
	numerics = pd.read_csv(IN_FILE)
	keys = ['z', 'u', 'beta', 'history', 'analytical']
	results = {key: [] for key in keys}

	# initialize wave conditions
	wave = fl.WaterWave(DEPTH, A, WAVELENGTH)
	k = wave.wavenum
	omega = wave.angular_freq
	analytical = False

	# analysis for negatively buoyant particles (beta < 1)
	warnings.filterwarnings('ignore')
	for beta, history in product(BETAS[1:], [True, False]):
		# extract and normalize data
		z_bar, u_bar = extract_data(['z_crossings', 'u_bar'], numerics,
									{'beta': beta, 'history': history})
		z_bar = z_bar.to_numpy()
		u_bar = u_bar.to_numpy()
		u_bar /= k * A
		extended_range = np.linspace(0, z_bar[-1], EXT_RANGE) if beta < 1 else \
						 np.linspace(-7, 0, EXT_RANGE)

		# fit a curve to the numerical data
		coefficients, _ = curve_fit(power, z_bar, u_bar, maxfev=MAXFEV)
		a, b, offset = coefficients
		u_bar = power(extended_range, a, b, offset)

		# store fitted curves
		results = update_results(results, [extended_range, u_bar],
								[beta, history, analytical])

	# compute and store analytical solutions
	beta, history, analytical = BETAS[0], None, True
	h = k * DEPTH
	z = np.linspace(0, -7, EXT_RANGE)
	u_d = np.cosh(2 * (z + h)) / (2 * np.sinh(h) ** 2)
	results = update_results(results, [z, u_d], [beta, history, analytical])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

def power(x, a, b, offset): return a * np.exp(b * x) + offset

if __name__ == '__main__':
	main()

import warnings
import numpy as np
import pandas as pd
import scipy.constants as constants
from scipy.optimize import curve_fit
from tqdm.contrib.itertools import product

from utils.data_tools import extract_data, update_results
from models import my_system as ts
from examples.water_wave.st_analysis.numerics import STOKES_NUMS, BETAS, \
													 WAVELENGTH, AMPLITUDE
from examples.water_wave.st_analysis.numerics import OUT_FILE as IN_FILE

EXT_RANGE = 300
MAXFEV=100000
OUT_FILE = '../../data/water_wave/st_analysis.csv'

def main():
	"""
	Fit curves to numerical drift velocity data of particles in a wave.

	Curves are fit to numerical solutions for the period-averaged Stokes drift
	velocity of particles of varying sizes (Stokes numbers). Results are saved
	to the `data/water_wave` directory.
	"""
	results = {'z_bar': [], 'u_bar': [], 'St': [], 'beta': [], 'history': []}

	# read data
	numerics = pd.read_csv(IN_FILE)
	k = 2 * np.pi / WAVELENGTH

	# analysis for negatively buoyant particles (beta < 1)
	warnings.filterwarnings('ignore')
	for stokes_num, beta, history in product(STOKES_NUMS, BETAS, [True, False]):
		params = {'St': stokes_num, 'beta': beta, 'history': history}
		z, u_bar = extract_data(['z', 'u_bar'], numerics, params)
		u_bar /= k * AMPLITUDE

		# perform curve fitting if there are numerical solutions
		z = z.to_numpy()
		if np.any(z):
			extended_range = np.linspace(0, -10, EXT_RANGE)
			f = lambda x, a, b, c : a + b * np.exp(c * x)

			# fit curve to data and store solutions
			coefficients, covariance = curve_fit(f, z, u_bar, maxfev=MAXFEV)
			a, b, c = coefficients
			u_bar = f(extended_range, a, b, c)
			results = update_results(results, [extended_range, u_bar],
									[stokes_num, beta, history])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

if __name__ == '__main__':
	main()

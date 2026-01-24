import warnings
import numpy as np
import pandas as pd
import scipy.constants as constants
from scipy.optimize import curve_fit
from tqdm.contrib.itertools import product

from utils.data_tools import extract_data, update_results, print_parameter, g
from models import my_system as ts
from examples.linear_wave.st_analysis.numerics import STOKES_HATS, RS, \
													  WAVELENGTH, AMPLITUDE
from examples.linear_wave.st_analysis.numerics import OUT_FILE as IN_FILE

EXT_RANGE = 300
MAXFEV= 100000
OUT_FILE = '../../data/linear_wave/st_analysis.csv'

def main():
	r"""
	Fit curves to numerical drift velocity data of particles in a wave.

	Curves are fit to numerical solutions for the period-averaged Stokes drift
	velocity of particles of varying sizes (Stokes numbers) for a negatively and
	a positively buoyant density ratio. The function used to fit the curve to
	the data is, $$g(x, A, \delta, \text{offset}) = Ae^{\delta x}
	+ \text{offset}.$$ Results are saved to the `data/linear_wave` directory.

	See Also
	--------
	utils.data_tools.g()
	"""
	results = {'z_bar': [], 'u_bar': [], 'Sthat': [], 'R': [], 'history': []}
	numerics = pd.read_csv(IN_FILE)
	warnings.filterwarnings('ignore')

	# analysis for negatively buoyant particles (beta < 1)
	for stokes_hat, r, history in product(STOKES_HATS, RS, [True, False]):
		params = {'Sthat': stokes_hat, 'R': r, 'history': history}
		z, u_bar = extract_data(['z', 'u_bar'], numerics, params)

		# perform curve fitting if there are numerical solutions
		z = z.to_numpy()
		if 3 <= len(z):
			extended_range = np.linspace(0, -10, EXT_RANGE)

			# fit curve to data and store solutions
			coefficients, covariance = curve_fit(g, z, u_bar, maxfev=MAXFEV)
			a, delta, offset = coefficients
			u_bar = g(extended_range, a, delta, offset)
			results = update_results(results, [extended_range, u_bar],
									[stokes_hat, r, history])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

if __name__ == '__main__':
	main()

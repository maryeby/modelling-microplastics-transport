import warnings
import numpy as np
import pandas as pd
import scipy.constants as constants
from scipy.optimize import curve_fit
from itertools import product

from utils.data_tools import extract_data, update_results, g
from models import my_system as ts
from examples.linear_wave.st_analysis.numerics import STOKES_HATS
from examples.linear_wave.st_analysis.numerics import RS as CONSTANT_RS
from examples.linear_wave.r_analysis.numerics import RS
from examples.linear_wave.r_analysis.numerics import STOKES_HAT as \
													 CONSTANT_ST_HAT
from examples.stokes_wave.h_analysis.numerics import DEPTHS
from examples.stokes_wave.h_analysis.numerics import OUT_FILE as IN_FILE

EXT_RANGE = 300
MAXFEV = 100000
KEYS = ['z_bar', 'u_d_bar', 'depth', 'Sthat', 'S', 'R', 'history']
OUT_FILE = '../../data/stokes_wave/h_analysis.csv'

def main():
	r"""
	Fit curves to numerical drift velocity data of particles in a wave.

	Curves are fit to numerical solutions for the period-averaged Stokes drift
	velocity of particles of varying sizes (Stokes numbers) and varying density
	ratios in water of varying depths. The expression $$g(x, A, \delta,
	\text{offset}) = Ae^{\delta x} + \text{offset}$$ is used to fit curves to
	the data. Results are saved to the `data/stokes_wave` directory.

	See Also
	--------
	utils.data_tools.g()
	"""
	results = {key: [] for key in KEYS}
	numerics = pd.read_csv(IN_FILE)
	warnings.filterwarnings('ignore')

	# create variables to use when extracting numerical data
	repeated_st = [STOKES_HATS[0], STOKES_HATS[3], STOKES_HATS[2]] \
				+ STOKES_HATS[:3] + [CONSTANT_ST_HAT] * len(RS[1:-1]) \
				* len(DEPTHS)
	repeated_rs = [CONSTANT_RS[0]] * 3 + [CONSTANT_RS[1]] * 3 + RS[1:-1]
	repeated_rs, _, _ = zip(*product(repeated_rs, DEPTHS, [True, False]))
	repeated_st, repeated_hs, repeated_history = zip(*product(repeated_st,
													 DEPTHS, [True, False]))
	loop_params = zip(repeated_st, repeated_rs, repeated_hs, repeated_history)

	# extract data and fit curves
	for stokes_hat, r, h, history in loop_params:
		names = ['z_bar', 'u_d_bar', 'Sthat/gamma']
		params = {'Sthat': stokes_hat, 'R': r, 'depth': h, 'history': history}
		z, u_bar, s = extract_data(names, numerics, params)
		z = z.to_numpy()

		# perform curve fitting if there are numerical solutions
		if 3 <= len(z):
			# fit curve to data and store solutions
			extended_range = np.linspace(0, -10, EXT_RANGE)
			coefficients, covariance = curve_fit(g, z, u_bar, maxfev=MAXFEV)
			a, delta, offset = coefficients
			u_bar = g(extended_range, a, delta, offset)
			results = update_results(results, [extended_range, u_bar],
									[h, stokes_hat, s.iloc[0], r, history])
		else:
			print(params, s)
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

if __name__ == '__main__':
	main()

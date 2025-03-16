import warnings
import numpy as np
import pandas as pd
import itertools
import scipy.constants as constants
from scipy.optimize import curve_fit

from utils.data_tools import extract_data, update_results
from models import water_wave as fl
from examples.water_wave.beta_analysis.numerics import AMPLITUDE as A
from examples.water_wave.beta_analysis.numerics import DEPTH, WAVELENGTH, BETAS

EXT_RANGE = 100
IN_FILE = '../../data/water_wave/beta_numerics.csv'
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
	solutions. For negatively buoyant particles, the drift velocity is averaged
	over each wave period, and single curve is fit to the numerical solutions of
	each simulation. Results are saved to the `data/water_wave` directory.

	See Also
	--------
	models.my_system.compute_drift_velocity

	References
	----------
	[^1]: [T. S. van den Bremer & Ø. Breivik (2018)](https://doi.org/10.1098/rsta.2017.0104)
		  Stokes drift. *Philosophical Transactions of the Royal Society A:
		  Mathematical, Physical and Engineering Sciences* 376(2111), 20170104.
	"""
	keys = ['z', 'u', 'beta', 'history', 'analytical']
	results = {key: [] for key in keys}
	numerics = pd.read_csv(IN_FILE)
	wave = fl.WaterWave(DEPTH, A, WAVELENGTH)
	k = wave.wavenum
	omega = wave.angular_freq
	analytical = False

	# analysis for negatively buoyant particles (beta < 1)
	warnings.filterwarnings('ignore')
	for beta, history in itertools.product(BETAS, [True, False]):
		# use a power law for data with history effects, exponential otherwise
		if history:
			f = lambda x, a, b, c, d : a * b ** (c * x + d)
		else:
			f = lambda x, a, b, c, d : a * np.exp(b * x) + c * np.exp(d * x)

		# extract and normalize data
		z_bar, u_bar = extract_data(['z_crossings', 'u_bar'], numerics,
									{'beta': beta, 'history': history})
		z_bar = z_bar.to_numpy()
		u_bar = u_bar.to_numpy()
		u_bar /= k * A

		# fit curve to data
		coefficients, covariance = curve_fit(f, z_bar, u_bar)
		a, b, c, d = coefficients
		extended_range = np.linspace(0, z_bar[-1], EXT_RANGE)
		u_bar = f(extended_range, a, b, c, d)

		# store estimated solutions
		results = update_results(results, [extended_range, u_bar],
								[beta, history, analytical])

	# compute and store analytical solutions
	beta, history, analytical = 1, None, True
	z = np.linspace(0, -DEPTH, EXT_RANGE) / k
	u_d = omega * A * A * k * np.cosh(2 * k * (z + DEPTH)) \
					   / (2 * np.sinh(k * DEPTH) ** 2)
	z *= k
	u_d /= omega * A * A * k
	results = update_results(results, [z, u_d], [beta, history, analytical])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

if __name__ == '__main__':
	main()

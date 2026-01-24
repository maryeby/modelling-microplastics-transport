import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import find_peaks
from tqdm.contrib.itertools import product

from utils.data_tools import extract_data, update_results
from utils.plot import initialize_figure as fig
from examples.linear_wave.forces.analysis import FORCES
from examples.linear_wave.forces.numerics import STOKES_HATS, RS
from examples.stokes_wave.forces.numerics import OUT_FILE as IN_FILE

KEYS = ['force', 'num_peaks', 'Sthat', 'St', 'R']
PLOT_NAME, PLOT_STHAT, PLOT_R = 'inertial_force', 0.14776, 0.66
OUT_FILE = '../../data/stokes_wave/spectral_analysis.csv'

def main():
	r"""
	Perform spectral analysis on the numerical forces data.

	Analysis is performed using the Fast Fourier Transform, and results are
	saved to the `data/stokes_wave` directory.
	"""
	# read data, suppress warnings, create Wave object
	numerics = pd.read_csv(IN_FILE)
	warnings.filterwarnings('ignore')
	results = {key: [] for key in KEYS}

	for sthat, r, name in product(STOKES_HATS, RS, FORCES):
		# extract numerical data
		params = {'Sthat': sthat, 'R': r}
		t = extract_data('t', numerics, params).to_numpy()
		if name == 'inertial_force':
			fpg, mass = extract_data(['fluid_pressure_gradient_x',
									  'added_mass_force_x'], numerics, params)
			force = fpg.to_numpy() + mass.to_numpy()
		elif name != 'xdot':
			name += '_x'
			force = extract_data(name, numerics, params).to_numpy()
			name = name[:-2]
		else:
			force = extract_data(name, numerics, params).to_numpy()

		# take the Fast Fourier Transform
		force_fft = fft(force)
		n = len(force_fft) // 2
		positive_fft = np.abs(force_fft[:n])

		# find the peaks
		peaks, sols = find_peaks(positive_fft, height=1)
		max_peak = np.round(np.max(sols['peak_heights']), 5) if 0 < len(peaks) \
				   else None
		tol = max_peak / 10

		# find the number of significant peaks (height > 0.1x the max peak)
		peaks, _ = find_peaks(positive_fft, height=tol)
		gamma = 1 / r - 1 / 2
		results = update_results(results, [], [name, len(peaks), sthat,
											   sthat * gamma, r])
		# plot example figure
		if name == PLOT_NAME and sthat == PLOT_STHAT and r == PLOT_R:
			fig('Hz', r'inertial force $|A|$')
			plt.plot(positive_fft, '-k')

	# write to data file
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)
	plt.show()

if __name__ == '__main__':
	main()

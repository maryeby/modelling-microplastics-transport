import warnings
import numpy as np
import pandas as pd
import itertools
import csv
import scipy.constants as constants

from scipy.optimize import curve_fit
from scipy.stats import linregress
from tqdm import tqdm

from utils.data_tools import extract_data, update_results
from models import water_wave as fl
from models.my_system import compute_drift_velocity as find_crossings
from examples.water_wave.forces.numerics import STOKES_NUMS, X_0, Z_0, DEPTH, \
	 AMPLITUDE, WAVELENGTH, BETA, DELTA_T, INCLUDE_HISTORY

ST_TO_SAVE = 0.1
NAMES = ['t', 'x', 'z', 'xdot', 'fluid_pressure_gradient_x',
		 'added_mass_force_x', 'stokes_drag_x', 'history_force_x']
KEYS = ['t', 'inertial', 'stokes_drag', 'history', 'velocity', 'R^2']
ROW_LABELS = KEYS[1:5]
COL_LABELS = ['A', 'delta', 'omega', 'phi', 'offset', 'R^2', 'St']
IN_FILE = '../../data/water_wave/forces_numerics.csv'
OUT_FILE1 = '../../data/water_wave/forces_coeffs.csv'
OUT_FILE2 = '../../data/water_wave/forces_curve_fit.csv'

def main():
	r"""
	Fit curves to numerical forces data and save the associated coefficients.

	For particles of different sizes (Stokes numbers) in a linear wave of
	deep water, a curve is fit to each of the forces over time. The coefficents
	resulting from the curve fitting are saved, and the curve data is saved for
	the particle with Stokes number `ST_TO_SAVE`. Results are saved to the
	`data/water_wave` directory.

	Notes
	-----
	The equation used to fit a curve to the data is,
	$$f = A * \exp(-\delta t) * \sin{(\omega t + \phi)}
					   + \text{offset},$$
	and the coefficients saved are the amplitude *A*, angular frequency *ω*,
	*δ*, phase shift *ϕ*, and offset.
	"""
	# read numerical data and write headers to OUT_FILE
	numerics = pd.read_csv(IN_FILE)
	file = open(OUT_FILE1, 'w')
	writer = csv.writer(file)
	writer.writerow([''] + COL_LABELS)
	file.close()

	# create variables for the curve fitting
	wave = fl.WaterWave(DEPTH, AMPLITUDE, WAVELENGTH)
	omega = wave.angular_freq
	coefficients = [0.06, omega, 0.7, 2, -0.037]
	fitted_forces = []

	warnings.filterwarnings('ignore')
	for stokes_num in STOKES_NUMS:
		# retrieve relevant numerical data
		data = extract_data(NAMES, numerics, {'St': stokes_num})
		data = [s.to_numpy() for s in data]
		t, x, z, xdot, fpg, mass, drag, history = data
		inertial = fpg + mass

		# initialize local variables for curve fitting
		forces = [xdot, inertial, drag, history]
		initial_guess = coefficients
		results_array = [[0] * len(COL_LABELS)] * len(ROW_LABELS)
		phi_u = None

		for i in range(len(forces)):
			# fit a curve to the force data
			force = forces[i]
			coefficients, covariance = curve_fit(f, t, force, p0=coefficients,
												 maxfev=100000)
			A, delta, _, phi, offset = coefficients
			coefficients[2] = omega
			curve = f(t, A, delta, omega, phi, offset)

			# compute phi relative to the velocity phase shift
			if i == 0: phi_u = phi
			phi -= phi_u

			# save the results if necessary and compute the R-value
			if stokes_num == ST_TO_SAVE: fitted_forces.append(curve)
			stats = linregress(force, curve)

			# restrict A to be positive
			if A < 0:
				phi += np.pi
				A = np.abs(A)
				coefficients[0] = A

			# map phi to [0, 2pi]
			if phi < 0: phi += 2 * np.pi
			if phi > 2 * np.pi: phi -= 2 * np.pi

			# store coefficients, R^2 value, and Stokes num in results array
			result_list = [A, delta, omega, phi, offset,
						   stats.rvalue * stats.rvalue, stokes_num]
			results_array[i] = result_list

		# save estimated solutions for specified Stokes nums
		if stokes_num == ST_TO_SAVE:
			results_dict = {key: [] for key in KEYS}
			results_dict = update_results(results_dict, [t, fitted_forces[1],
										  fitted_forces[2], fitted_forces[3],
										  fitted_forces[0]],
										 [stats.rvalue * stats.rvalue])

		# once results array is full, convert to dataframe and write to file
		pd.DataFrame(results_array, index=ROW_LABELS, columns=COL_LABELS)\
		  .to_csv(OUT_FILE1, header=False, mode='a')
	pd.DataFrame(results_dict).to_csv(OUT_FILE2, index=False)

def f(t, A, delta, omega, phi, offset):
	return A * np.exp(-delta * t) * np.sin(omega * t + phi) + offset

if __name__ == '__main__':
	main()

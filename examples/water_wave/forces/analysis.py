import warnings
import numpy as np
import pandas as pd
import scipy as scp
from matplotlib import pyplot as plt
from tqdm.contrib.itertools import product

from utils.data_tools import extract_data
from utils.plot import initialize_figure as fig
from models import water_wave as fl
from examples.water_wave.forces.numerics import STOKES_NUMS, RS, DEPTH, \
	 AMPLITUDES, WAVELENGTH, DELTA_T, NUM_PERIODS, EPSILONS
from examples.water_wave.forces.numerics import OUT_FILE as IN_FILE

# labels for plotting and storing solutions
FORCES = ['inertial_force', 'stokes_drag', 'history_force', 'xdot']
LABELS = ['inertial forces', 'Stokes drag', 'history force',
		  'particle velocity']
COEFFS = ['A', 'delta', 'phi', 'offset', 'R^2']
COLUMNS = ['force'] + COEFFS + ['St', 'R', 'epsilon']

ST_TO_SHOW = STOKES_NUMS[1]
R_TO_SHOW = RS[4]
EPSILON_TO_SHOW = EPSILONS[1]
A_TO_SHOW = AMPLITUDES[1]
MAXFEV = 500000
OUT_FILE = '../../data/water_wave/forces_coeffs.csv'

def main():
	r"""
	Fit curves to numerical forces data and save the associated coefficients.

	For particles of different sizes (Stokes numbers) in a linear wave of
	deep water, a curve is fit to each of the forces over time. The coefficents
	resulting from the curve fittings are saved, and the curves are plotted for
	the particle with Stokes number `ST_TO_SHOW`. Results are saved to the
	`data/water_wave` directory.

	Notes
	-----
	The equation used to fit a curve to the data is,
	$$f = A \exp(-\delta t) \sin{(\omega t + \phi)} + \text{offset},$$
	and the coefficients saved are the amplitude *A*, angular frequency
	$\omega$, decay rate $\delta$, phase shift $\phi$, and offset.
	"""
	# read data, suppress warnings, create empty list to store results
	numerics = pd.read_csv(IN_FILE)
	warnings.filterwarnings('ignore')
	coeff_results = []

	for stokes_num, r, amplitude, name in product(STOKES_NUMS, RS, AMPLITUDES,
												  FORCES):
		# create Wave object and set variables for curve fitting
		wave = fl.WaterWave(DEPTH, amplitude, WAVELENGTH)
		epsilon = np.round(wave.wavenum * amplitude, 3)
#		if stokes_num == STOKES_NUMS[0]:
		initial_guess1 = [0.07, 0.01, epsilon, 5e-3, 2e-4]

		# extract numerical data
		params = {'St': stokes_num, 'R': r, 'epsilon': epsilon}
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

		# compute coefficients using SciPy curve fit
		t = np.arange(0, wave.period * NUM_PERIODS, DELTA_T)
		if len(t) != len(force): t = t[:len(force)]
		if name != 'inertial_force':
			coefficients, _ = scp.optimize.curve_fit(f, t, force,
								  p0=[0, 0.22, epsilon, 0 ,0], maxfev=MAXFEV)
		else:
			if len(t) == 0: print(params, name)
			coefficients, _ = scp.optimize.curve_fit(f, t, force,
										   p0=initial_guess1, maxfev=MAXFEV)
		a, delta, _, phi, offset = coefficients
		coefficients[2] = epsilon
		coefficients = coefficients.tolist()
#		initial_guess1 = coefficients
		initial_guess2 = coefficients[:2] + [coefficients[-1]]
		del coefficients[2]

		# fit curve, compute R^2 value, and store results
		fitted_curve = f(t, a, delta, epsilon, phi, offset)
		stats = scp.stats.linregress(force, fitted_curve)
		rsq = stats.rvalue * stats.rvalue
		coefficients.append(rsq)
		coeff_results.append([name] + coefficients + [stokes_num, r, epsilon])

		# plot ST_TO_SHOW
		if stokes_num == ST_TO_SHOW and r == R_TO_SHOW \
									and epsilon == EPSILON_TO_SHOW:
			i = FORCES.index(name)
			x_label = 't' if name == 'xdot' else None
			fig(x_label, LABELS[i], 411 + i)
			plt.scatter(t, force, marker='.', edgecolors='k', facecolors='none')
			plt.plot(t, fitted_curve, '--k')
			if i == 0:
				plt.suptitle(rf'$St$ = {ST_TO_SHOW:.3f},'
				+ rf'$R$ = {R_TO_SHOW:.2f}, $\epsilon$ = {EPSILON_TO_SHOW:.2f}')
	# restrict coefficients and write to data file
	coeffs = pd.DataFrame(coeff_results, columns=COLUMNS)
	coeffs = restrict_coeffs(coeffs)
	coeffs.round(6).to_csv(OUT_FILE, index=False)
	plt.show()

def f(t, a, delta, epsilon, phi, offset):
	return a * np.exp(delta * t) * np.sin(t / epsilon + phi) + offset

def restrict_coeffs(df):
	r"""Restrict *A* to be positive and map $\phi$ to $[0, 2\pi]$ in `df`."""
	# force A to be positive and shift the phase by pi where necessary
	a, phi = df['A'].to_numpy(), df['phi'].to_numpy()
	phi[a < 0] += np.pi
	a = np.abs(a)
	df.replace(df['A'].tolist(), a.tolist(), inplace=True)
	df.replace(df['phi'].tolist(), phi, inplace=True)

	# map velocity phase to [0, 2pi]
	phi_xdot = extract_data('phi', df, {'force': 'xdot'}).to_numpy()
	phi_xdot %= 2 * np.pi

	# ensure phi is relative to the velocity phase, map to the range [0, 2pi]
	vel_phi = []
	for i in range(0, len(phi_xdot), 2):
		vel_phi += [phi_xdot[i], phi_xdot[i + 1]] * len(FORCES)
	phi -= np.array(vel_phi)
	phi %= 2 * np.pi

	# organize and return results
	df.replace(df['phi'].tolist(), phi, inplace=True)
	return df

if __name__ == '__main__':
	main()

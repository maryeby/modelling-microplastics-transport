import warnings
import numpy as np
import pandas as pd
import scipy as scp
from matplotlib import pyplot as plt
from tqdm.contrib.itertools import product

from utils.data_tools import extract_data, print_characteristic_params
from utils.plot import initialize_figure as fig
from transport_framework import particle as prt
from models import linear_wave as fl
from models import my_system as ts
from examples.linear_wave.forces.numerics import STOKES_HATS, RS, DEPTH, \
	 AMPLITUDE, WAVELENGTH, DELTA_T, NUM_PERIODS
from examples.linear_wave.forces.numerics import OUT_FILE as IN_FILE

# labels for plotting and storing solutions
FORCES = ['inertial_force', 'stokes_drag', 'history_force', 'xdot']
LABELS = ['inertial forces', 'Stokes drag', 'history force',
		  'particle velocity']
COEFFS = ['A', 'delta', 'phi', 'offset', 'R^2']
COLUMNS = ['force'] + COEFFS + ['Sthat', 'St', 'R']

R_TO_SHOW = RS[-1]
ST_TO_SHOW = np.round(STOKES_HATS[9] * (1 / R_TO_SHOW - 0.5), 5)
MAXFEV = 10000
OUT_FILE = '../../data/linear_wave/forces_coeffs.csv'

def main():
	r"""
	Fit curves to numerical forces data and save the associated coefficients.

	For particles of different sizes (Stokes numbers) in a linear wave of
	deep water, a curve is fit to each of the forces over time. The coefficents
	resulting from the curve fittings are saved, and the curves are plotted for
	the particle with Stokes number `ST_TO_SHOW` and density ratio `R_TO_SHOW`.
	Results are saved to the `data/linear_wave` directory.

	Notes
	-----
	The equation used to fit a curve to the data is,
	$$f(t) = A e^{-\delta t} \sin{(\omega t + \phi)} + \text{offset},$$
	and the coefficients saved are the amplitude *A*, angular frequency
	$\omega$, decay rate $\delta$, phase shift $\phi$, and offset.
	"""
	# read data, suppress warnings
	numerics = pd.read_csv(IN_FILE)
	warnings.filterwarnings('ignore')

	# initialize Wave object, set initial guesses
	wave = fl.LinearWave(DEPTH, AMPLITUDE, WAVELENGTH)
	coeff_results = []
	guesses = [[0] * 4] * len(FORCES)
	guesses[1][1] = 0.22

	for stokes_hat, r, name in product(STOKES_HATS, RS, FORCES):
		# create objects and set variables for curve fitting
		particle = prt.Particle(stokes_hat)
		system = ts.MyTransportSystem(particle, wave, r)
		i = FORCES.index(name)
		guess = guesses[i]

		# extract numerical data
		params = {'Sthat': stokes_hat, 'R': r}
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

		# compute coefficients using SciPy curve fit
		coefficients, _ = scp.optimize.curve_fit(f, t, force, p0=guess,
												 maxfev=MAXFEV)
		a, delta, phi, offset = coefficients
		coefficients = coefficients.tolist()
		guesses[i] = coefficients.copy()

		# fit curve, compute R^2 value, and store results
		fitted_curve = f(t, a, delta, phi, offset)
		stats = scp.stats.linregress(force, fitted_curve)
		rsq = stats.rvalue * stats.rvalue
		coefficients.append(rsq)
		coeff_results.append([name] + coefficients + [stokes_hat,
													  system.stokes_num, r])
		# plot ST_TO_SHOW
		if system.stokes_num == ST_TO_SHOW and r == R_TO_SHOW:
			x_label = 't' if name == 'xdot' else None
			fig(x_label, LABELS[i], 411 + i)
			plt.scatter(t, force, marker='.', edgecolors='k', facecolors='none')
			plt.plot(t, fitted_curve, '--k')
			if i == 3: print_characteristic_params(wave, particle, system)
			if i == 0:
				plt.suptitle(rf'$St$ = {ST_TO_SHOW:.3f}, $R$ = {R_TO_SHOW:.2f}')

	# restrict coefficients and write to data file
	coeffs = pd.DataFrame(coeff_results, columns=COLUMNS)
	coeffs = restrict_coeffs(coeffs)
	coeffs.round(6).to_csv(OUT_FILE, index=False)
	plt.show()

def f(t, a, delta, phi, offset):
	return a * np.exp(delta * t) * np.sin(t + phi) + offset

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
	for i in range(len(phi_xdot)): vel_phi += [phi_xdot[i]] * len(FORCES)
	phi -= np.array(vel_phi)
	phi %= 2 * np.pi

	# organize and return results
	df.replace(df['phi'].tolist(), phi, inplace=True)
	return df

if __name__ == '__main__':
	main()

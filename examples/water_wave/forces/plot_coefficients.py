import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data
from examples.water_wave.forces.numerics import STOKES_NUMS
from examples.water_wave.forces.analysis import FORCES, COEFFS, METHODS, \
												ST_TO_SHOW

STYLES = ['-', '--', '-.', ':']
IN_FILE1 = '../../data/water_wave/forces_numerics.csv'
IN_FILE2 = '../../data/water_wave/forces_coeffs.csv'
#IN_FILE1 = '../../data/water_wave/st_star/numerics8.csv'
#IN_FILE2 = '../../data/water_wave/st_star/coeffs8.csv'
TOL = 0.96

def main():
	"""
	Plot forces over time with curves fit to the data, and coefficients vs *St*.

	For curves fit to the forces acting on particles of different sizes (Stokes
	numbers) in a linear wave of deep water, the resulting coefficents are
	plotted over the Stokes number.
	"""
	# read data and get time series
	numerics = pd.read_csv(IN_FILE1)
	coefficients = pd.read_csv(IN_FILE2)

	# coefficients from curve fit
#	plt.suptitle('SciPy curve fit')
#	for i in range(len(COEFFS[:-1])):
#		generate_coeff_subplot(i, COEFFS[i], METHODS[0], coefficients)
	generate_coeff_subplot(0, COEFFS[0], METHODS[0], coefficients)
	generate_coeff_subplot(1, COEFFS[2], METHODS[0], coefficients)

	# coefficients from hilbert transform
#	plt.suptitle('Hilbert transform')
#	for i in range(len(COEFFS[:-1])):
#		generate_coeff_subplot(i, COEFFS[i], METHODS[1], coefficients)
	plt.show()

def generate_coeff_subplot(plot_num, coeff, method, coefficients):
	"""Plot *St* vs a coefficient from the `coefficients` `DataFrame`."""
	x = r'$St$' #if coeff == 'phi' or coeff == 'offset' else None
	y = rf'$\{coeff}$' if coeff == 'phi' or coeff == 'delta' else coeff
	y = rf'${coeff}$' if coeff == 'A' else y
#	fig(x, y, 221 + plot_num, make_square=True)
	fig(x, y, 121 + plot_num, make_square=False, add_subplot_labels=True)
	if coeff == 'phi': plt.yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2,
								   2 * np.pi], ['0', r'$\frac{\pi}{2}$',
								   r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
	plt.axvline(ST_TO_SHOW, c='silver')
	for i in range(len(FORCES[:-1])):
		plot_points(FORCES[i], coeff, method, STYLES[i + 1], coefficients)
#	for i in range(len(FORCES)):
#		plot_points(FORCES[i], coeff, method, STYLES[i], coefficients)

def plot_points(force, coeff, method, ls, coefficients):
	"""Plot points from the `coefficients` `DataFrame`."""
	fmt = ls + 'k'
	params = {'force': force, 'method': method}
	names = [coeff, 'R^2', 'St']
	label = force if coeff == 'phi' else ''

	# plot the Stokes numbers vs the value of the coefficient
	data, rsq, st = extract_data(names, coefficients, params)
	plt.plot(STOKES_NUMS, data, fmt, label=label)

	# plot quality control points
	qc_st = st.where(rsq < TOL).dropna()
	qc_data = data.where(rsq < TOL).dropna()
	label = rf'$R^2 < {TOL}$' if coeff == 'phi' and force == 'history_force' \
							  else ''
#	plt.scatter(qc_st[3:], qc_data[3:], edgecolors='k', facecolors='none',
#			    label=label)
#	if coeff == 'phi':
#		plt.legend()
#		max_os = np.max(np.abs(extract_data('offset', coefficients,
#										   {'method': METHODS[0]}).to_numpy()))
#		max_os_a = extract_data('A', coefficients, {'offset': max_os}).iloc[0]
#		percent_os = max_os * 100 / max_os_a
#		print(f'max offset: {max_os:.4f}, {percent_os:.2f}% of amplitude',
#			  f'{max_os_a:.4f}')


if __name__ == '__main__':
	main()

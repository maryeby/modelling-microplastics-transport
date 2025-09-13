import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data
from examples.water_wave.forces.numerics import STOKES_NUMS
from examples.water_wave.forces.analysis import FORCES, LABELS, COEFFS, \
												ST_TO_SHOW
from examples.water_wave.forces.analysis import R_TO_SHOW as R
from examples.water_wave.forces.analysis import EPSILON_TO_SHOW as EPSILON

FORMATS = ['-k', '--k', '-.k', ':k']
XLABEL = r'$St$'
YLABELS = [r'$A$', r'$\delta$', r'$\phi$', 'offset']
IN_FILE1 = '../../data/water_wave/forces_numerics.csv'
IN_FILE2 = '../../data/water_wave/forces_coeffs.csv'
TOL = 0.96

def main():
	"""
	Plot forces over time with curves fit to the data, and coefficients vs *St*.

	For curves fit to the forces acting on particles of different sizes (Stokes
	numbers) in a linear wave of deep water, the resulting coefficents are
	plotted over the Stokes number.
	"""
	# read data
	numerics = pd.read_csv(IN_FILE1)
	coefficients = pd.read_csv(IN_FILE2)

	# create plots
	for i in range(len(COEFFS) - 1):
		fig(XLABEL, YLABELS[i], 221 + i)
		plt.axvline(ST_TO_SHOW, c='silver')
		if COEFFS[i] == 'phi':
			f = len(FORCES[:-1])
			plt.ylim(0, 2 * np.pi)
			plt.yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
					   ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$',
						r'$2\pi$'])
		else:
			f = len(FORCES)
		for j in range(f):
			plot_points(FORCES[j], LABELS[j], COEFFS[i], FORMATS[j],
						coefficients)
	plt.show()

def plot_points(force, label, coeff, fmt, coefficients):
	"""Plot points from the `coefficients` `DataFrame`."""
	names = [coeff, 'R^2', 'St']
	include_legend = force == FORCES[-1] and coeff == COEFFS[-2]

	# plot the Stokes numbers vs the value of the coefficient
	params = {'force': force, 'R': R, 'epsilon': EPSILON}
	data, rsq, st = extract_data(names, coefficients, params)
	plt.plot(STOKES_NUMS, data, fmt, label=label)

	# plot quality control points
	qc_st = st.where(rsq < TOL).dropna()
	qc_data = data.where(rsq < TOL).dropna()
	label = rf'$R^2 < {TOL}$' if include_legend else ''
	plt.scatter(qc_st[3:], qc_data[3:], edgecolors='k', facecolors='none',
			    label=label)

	# compute max offset and its % of the amplitude, add legend
	if include_legend:
		params = {'R': R, 'epsilon': EPSILON}
		max_os = np.max(np.abs(extract_data('offset', coefficients,
				 params).to_numpy()))
		max_os_a = extract_data('A', coefficients, {'offset': max_os})
		max_os_a = extract_data('A', coefficients, {'offset': -max_os}).iloc[0]\
				   if max_os_a.empty else max_os_a.iloc[0]
		percent_os = max_os * 100 / max_os_a
		print(f'max offset: {max_os:.4f}, {percent_os:.2f}% of amplitude',
			  f'{max_os_a:.4f}')
		plt.legend()

if __name__ == '__main__':
	main()

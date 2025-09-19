import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from utils.plot import initialize_figure as fig 
from utils.data_tools import extract_data
from utils.colors import COLORS
from examples.water_wave.shear.numerics import X_0S, RADIUS, ANGLE, Z_NEGATIVE,\
	 Z_POSITIVE, ST_TO_SHOW, BETA_TO_SHOW, STOKES_NUMS, BETAS, NUM_POINTS, \
	 WAVELENGTH, AMPLITUDE
from examples.water_wave.shear.numerics import SCALE as R_SCALE
from examples.water_wave.shear.numerics import OUT_FILE as IN_FILE1
from examples.water_wave.shear.analysis import OUT_FILE2 as IN_FILE2

ST_SCALE = WAVELENGTH / (2 * np.pi * AMPLITUDE) \
					  * (3 / (2 * BETA_TO_SHOW) - 1 / 2)

def main():
	"""Plot particle positions at regular intervals and other shear results."""
	numerics = pd.read_csv(IN_FILE1)
	shear_diff = pd.read_csv(IN_FILE2)
	point_index = range(NUM_POINTS)
	z_center = Z_NEGATIVE * RADIUS
	z_0s = np.round(RADIUS * np.sin(ANGLE) + z_center, 3)

	fig(r'$x$', r'$z$', equal_aspect=True, make_square=True, width='jfm')
	for i, history in product(point_index, [False, True]):
		params = {'x_0': X_0S[i], 'z_0': z_0s[i], 'St': ST_TO_SHOW,
				  'beta': BETA_TO_SHOW, 'history': history}
		mk = 'x' if history else '.'
		x, z = extract_data(['x', 'z'], numerics, params)
		plt.scatter(x, z, c=COLORS[i], marker=mk)

	fig('point index', 'S', width='jfm')
	for i, history in product(point_index, [False, True]):
		params = {'x_0': X_0S[i], 'z_0': z_0s[i], 'St': ST_TO_SHOW,
				  'beta': BETA_TO_SHOW, 'history': history}
		mk = 'x' if history else '.'
		total_shear = extract_data('shear', numerics, params).to_numpy()[-1]
		plt.scatter(i, total_shear, c=COLORS[i], marker=mk)

	fig(r'$x$', 'shear', make_square=True, width='jfm')
	for i, history in product(point_index, [False, True]):
		params = {'x_0': X_0S[i], 'z_0': z_0s[i], 'St': ST_TO_SHOW,
				  'beta': BETA_TO_SHOW, 'history': history}
		mk = 'x' if history else '.'
		ls = ':' if history else '-'
		x, shear = extract_data(['x', 'shear'], numerics, params)
		plt.plot(x[1:], shear[:-1], c=COLORS[i], marker=mk, ls=ls)

	fig(r'$St$', r'$\Delta\bar{S}$', 121, width='jfm', add_subplot_labels=True)
	stokes_num, shear = extract_data(['St', 'shear_difference'], shear_diff,
									 {'beta': BETA_TO_SHOW})
	plt.plot(stokes_num * ST_SCALE, shear, '-k')
	fig(r'$R$', num=122, width='jfm', add_subplot_labels=True)
	idx = shear_diff['beta'].sort_values().index
	beta, shear = extract_data(['beta', 'shear_difference'],
								shear_diff.loc[idx], {'St': ST_TO_SHOW})
	n = np.where(beta == 1)[0][0]
	plt.plot(beta[:n] * R_SCALE, shear[:n], '-k')
	plt.plot(beta[n:-6] * R_SCALE, shear[n:-6], '--k')
	plt.axvline(R_SCALE, c='silver')
	plt.show()

if __name__ == '__main__':
	main()

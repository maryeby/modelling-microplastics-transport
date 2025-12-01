import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from utils.plot import initialize_figure as fig 
from utils.data_tools import extract_data
from utils.colors import COLORS
from examples.linear_wave.shear.numerics import X_0S, RADIUS, ANGLE, \
	 Z_NEGATIVE, Z_POSITIVE, ST_TO_SHOW, R_TO_SHOW, STOKES_HATS, RS, \
	 NUM_POINTS, WAVELENGTH, AMPLITUDE
from examples.linear_wave.shear.numerics import OUT_FILE as IN_FILE1
from examples.linear_wave.shear.analysis import OUT_FILE2 as IN_FILE2

NEUTRAL_R = np.round(2 / 3, 5)

def main():
	"""Plot particle positions at regular intervals and other shear results."""
	numerics = pd.read_csv(IN_FILE1)
	shear_diff = pd.read_csv(IN_FILE2)
	point_index = range(NUM_POINTS)
	z_center = Z_NEGATIVE * RADIUS
	z_0s = np.round(RADIUS * np.sin(ANGLE) + z_center, 3)

	fig(r'$x$', r'$z$', equal_aspect=True, make_square=True, width='jfm')
	for i, history in product(point_index, [False, True]):
		params = {'x_0': X_0S[i], 'z_0': z_0s[i], 'Sthat': ST_TO_SHOW,
				  'R': R_TO_SHOW, 'history': history}
		mk = 'x' if history else '.'
		x, z = extract_data(['x', 'z'], numerics, params)
		plt.scatter(x, z, c=COLORS[i], marker=mk)

	fig('point index', '$M$', width='jfm')
	for i, history in product(point_index, [False, True]):
		params = {'x_0': X_0S[i], 'z_0': z_0s[i], 'Sthat': ST_TO_SHOW,
				  'R': R_TO_SHOW, 'history': history}
		mk = 'x' if history else '.'
		total_shear = extract_data('shear', numerics, params).to_numpy()[-1]
		plt.scatter(i, total_shear, c=COLORS[i], marker=mk)

	fig(r'$x$', 'shear', make_square=True, width='jfm')
	for i, history in product(point_index, [False, True]):
		params = {'x_0': X_0S[i], 'z_0': z_0s[i], 'Sthat': ST_TO_SHOW,
				  'R': R_TO_SHOW, 'history': history}
		mk = 'x' if history else '.'
		ls = ':' if history else '-'
		x, shear = extract_data(['x', 'shear'], numerics, params)
		plt.plot(x[1:], shear[:-1], c=COLORS[i], marker=mk, ls=ls)

	fig(r'$St$', r'$\Delta\overline{M}$', 121, width='jfm',
		add_subplot_labels=True)
	stokes_num, shear = extract_data(['Sthat', 'shear_difference'], shear_diff,
									 {'R': R_TO_SHOW})
	plt.plot(stokes_num, shear, '-k')
	fig(r'$R$', num=122, width='jfm', add_subplot_labels=True)
	idx = shear_diff['R'].sort_values().index
	r, shear = extract_data(['R', 'shear_difference'],
							 shear_diff.loc[idx], {'Sthat': ST_TO_SHOW})
	n = np.where(r == NEUTRAL_R)[0][0]
	plt.plot(r[:n], shear[:n], '-k')
	plt.plot(r[n:-6], shear[n:-6], '--k')
	plt.axvline(NEUTRAL_R, c='silver')
	plt.show()

if __name__ == '__main__':
	main()

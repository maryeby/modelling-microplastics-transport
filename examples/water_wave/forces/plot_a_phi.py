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

K = 2.07 # used to estimate the ratio of history to Stokes drag
TOL = 0.96
SCALE = 1 / EPSILON * (1 / R - 1 / 2)
STYLES = ['-.', '-', '--', ':']
XLABEL = r'$St$'
PHI_TICKS = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, 2 * np.pi / 3,
				3 * np.pi / 4, 5 * np.pi / 6, np.pi, 7 * np.pi / 6,
				5 * np.pi / 4, 4 * np.pi / 3, 3 * np.pi / 2, 5 * np.pi / 3,
				7 * np.pi / 4, 11 * np.pi / 6, 2 * np.pi]
PHI_LABELS = ['0', r'$\frac{\pi}{6}$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{3}$',
			 r'$\frac{\pi}{2}$', r'$\frac{2\pi}{3}$', r'$\frac{3\pi}{4}$',
			 r'$\frac{5\pi}{6}$', r'$\pi$', r'$\frac{7\pi}{6}$',
			 r'$\frac{5\pi}{4}$', r'$\frac{4\pi}{3}$', r'$\frac{3\pi}{2}$',
			 r'$\frac{5\pi}{3}$', r'$\frac{7\pi}{4}$', r'$\frac{11\pi}{6}$',
			 r'$\2pi$']
IN_FILE1 = '../../data/water_wave/forces_numerics.csv'
IN_FILE2 = '../../data/water_wave/forces_coeffs.csv'

def main():
	r"""
	Plot forces over time with curves fit to the data, and coefficients vs *St*.

	For curves fit to the drag forces acting on particles of different sizes
	(Stokes numbers) in a linear wave of deep water, the resulting *A* and
	$\phi$ coefficents are plotted over the Stokes number. The ratio between the
	drag forces is also plotted.
	"""
	# read and extract data
	numerics = pd.read_csv(IN_FILE1)
	coefficients = pd.read_csv(IN_FILE2)
	names = [COEFFS[0], 'R^2', 'St']
	params = {'force': FORCES[1], 'R': R, 'epsilon': EPSILON}
	stokes_drag, rsq, st = extract_data(names, coefficients, params)
	sthat = st.to_numpy() / EPSILON
	
	# plot St vs A
	fig(XLABEL, r'$A$', 121, add_subplot_labels=True, width='jfm')
	plt.xlim(0, 1)
	plt.axvline(ST_TO_SHOW * SCALE, c='silver')
	plt.plot(st * SCALE, stokes_drag, '--k')
	params['force'] = FORCES[2]
	history, rsq, st = extract_data(names, coefficients, params)
	plt.plot(st * SCALE, history, ':k')

	# plot St vs phi
	names[0] = COEFFS[2]
	params['force'] = FORCES[1]
	data, rsq, st = extract_data(names, coefficients, params)
	fig(XLABEL, r'$\phi$', 122, add_subplot_labels=True, width='jfm')
	plt.xlim(0, 1)
	plt.yticks(PHI_TICKS, PHI_LABELS)
	plt.axvline(ST_TO_SHOW * SCALE, c='silver')
	plt.plot(st * SCALE, data, '--k')
	params['force'] = FORCES[2]
	data, rsq, st = extract_data(names, coefficients, params)
	plt.plot(st * SCALE, data, ':k')

	# plot the ratio of drag forces
	stokes_drag, history = stokes_drag.to_numpy(), history.to_numpy()
	fig(r'$\widehat{St}$', 'History force / Stokes drag', make_square=True,
		width='jfm')
	plt.scatter(sthat, history / stokes_drag, marker='.', ec='k', fc='none')
	plt.plot(sthat, K * np.sqrt(sthat), ':k')
	plt.show()

if __name__ == '__main__':
	main()

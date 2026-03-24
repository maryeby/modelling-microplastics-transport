import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from utils.plot import plot_quality_control
from utils.data_tools import extract_data
from examples.linear_wave.forces.numerics import STOKES_HATS
from examples.linear_wave.forces.numerics import OUT_FILE as IN_FILE1
from examples.linear_wave.forces.analysis import FORCES, LABELS, COEFFS, \
												 ST_TO_SHOW
from examples.linear_wave.forces.analysis import R_TO_SHOW as R
from examples.linear_wave.forces.analysis import OUT_FILE as IN_FILE2

K = 2 # used to estimate the ratio of history to Stokes drag
Y = np.linspace(0, 1, 100)
PLOT_NUMS, FORMATS = [121, 122], ['--k', ':k']
XLABEL, YLABELS = r'$St$', [r'$A$', r'$\phi$']
XLIM = (0.15, 1)
PHI_TICKS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4,
			 3 * np.pi / 2, 7 * np.pi / 4, 2 * np.pi]
PHI_LABELS = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
			 r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$',
			 r'$\frac{7\pi}{4}$', r'$2\pi$']

def main():
	r"""
	Plot the Stokes numbers vs the amplitude and phase angle coefficients.

	For curves fit to the drag forces acting on particles of different sizes
	(Stokes numbers) in a linear wave of deep water, the resulting $A$ and
	$\phi$ coefficents are plotted over the Stokes number. The ratio between the
	drag forces is also plotted.
	"""
	# read data
	numerics = pd.read_csv(IN_FILE1)
	coefficients = pd.read_csv(IN_FILE2)

	# initialize and format each subplot
	for ylabel, num, coeff in zip(YLABELS, PLOT_NUMS, COEFFS[:3:2]):
		fig(XLABEL, ylabel, num, add_subplot_labels=True)
		plt.xlim(XLIM)
		plt.axvline(ST_TO_SHOW, c='silver')
		names = [coeff, 'R^2', 'Sthat', 'St']
		if num == PLOT_NUMS[1]:
			plt.ylim(np.pi / 3, PHI_TICKS[3])
			plt.yticks(PHI_TICKS[2:4], PHI_LABELS[2:4])

		# extract and plot data
		for force, fmt in zip(FORCES[1:3], FORMATS):
			params = {'force': force, 'R': R}
			data, rsq, sthat, st = extract_data(names, coefficients, params)
			plot_quality_control(st, data, rsq)
			plt.plot(st, data, fmt)

	# extract the drag forces
	x = np.linspace(0, 1, 100)
	stokes_drag = extract_data(COEFFS[0], coefficients, {'force': FORCES[1],
														 'R': R}).to_numpy()
	history = extract_data(COEFFS[0], coefficients, {'force': FORCES[2],
													 'R': R}).to_numpy()
	# plot the ratio of drag forces
	fig(r'$\chi$', r'$\widehat{St}$', make_square=True)
	plt.scatter(history / stokes_drag, sthat, marker='.', ec='k', fc='none')
	plt.plot(K * np.sqrt(Y), Y, ':k')
	plt.show()

if __name__ == '__main__':
	main()

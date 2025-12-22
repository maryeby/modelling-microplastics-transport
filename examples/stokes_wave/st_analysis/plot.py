import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data, print_parameter
from examples.linear_wave.st_analysis.numerics import STOKES_HATS, RS, \
													  WAVELENGTH
from examples.stokes_wave.st_analysis.numerics import AMPLITUDE
from examples.stokes_wave.st_analysis.numerics import OUT_FILE as IN_FILE1
from examples.stokes_wave.st_analysis.analysis import OUT_FILE as IN_FILE2

# position bubble labels
TEXT_POSITION_X = [-0.25, -0.398, -0.485, -0.304, -0.352]
TEXT_POSITION_Y = [-0.55, -0.79, -1.11, -0.81, -1.314]
PROPERTIES = dict(boxstyle='circle', fc='w', ec='k')
FS = 8

def main():
	"""
	Plot the drift velocity vs vertical position of particles in a wave.

	The numerical horizontal drift velocity and vertical position are averaged
	over each wave period. Solutions with and without history effects are
	included. Subplots are separated into negatively buoyant *(a)* and
	positively buoyant *(b)* cases.
	"""
	numerics = pd.read_csv(IN_FILE1)
	analysis = pd.read_csv(IN_FILE2)
	print_parameter('epsilon', 2 * np.pi / WAVELENGTH * AMPLITUDE)
	print_parameter('R', RS[0])
	print_parameter('R', RS[1])

	lims = None
	for i in range(len(RS)):
		# define variables depending on R
		if i == 0:
			lims = [-0.5, 0.05, -2, -0.4]
			nums = [STOKES_HATS[0]] + STOKES_HATS[-2:]
			ylabel = r'$\bar{z}$'
			hide_yticks = False
		else:
			lims = [-0.53, -0.2, -2, -0.4]
			nums = STOKES_HATS[:3]
			ylabel = None
			hide_yticks = True

		# initialize subplot and create subplot labels
		fig(r'$\bar{u}/\epsilon^2$', ylabel, 121 + i, make_square=True,
			add_subplot_labels=True, lims=lims, hide_yticks=hide_yticks)

		# plot curves and data points
		for stokes_hat, history in product(nums, [False, True]):
			# plot numerical solutions
			names = ['z', 'u_bar', 'Sthat/gamma']
			params = {'Sthat': stokes_hat, 'R': RS[i], 'history': history}
			z_bar, u_bar, sthat_gamma = extract_data(names, numerics, params)
			plt.scatter(u_bar, z_bar, marker='.', ec='k', fc='none')

			# plot fitted curves
			ls = '--' if history else '-'
			z_bar, u_bar = extract_data(['z_bar', 'u_bar'], analysis, params)
			plt.plot(u_bar, z_bar, c='k', ls=ls)

			# plot labels
			j = STOKES_HATS.index(stokes_hat)
			plt.text(TEXT_POSITION_X[j], TEXT_POSITION_Y[j],
					 f'{sthat_gamma.iloc[0]:.2g}', bbox=PROPERTIES, fontsize=FS)
	plt.show()

if __name__ == '__main__':
	main()

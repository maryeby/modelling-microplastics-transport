import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data
from examples.linear_wave.st_analysis.numerics import STOKES_HATS, RS, \
													  AMPLITUDE, WAVELENGTH
from examples.linear_wave.st_analysis.numerics import OUT_FILE as IN_FILE1
from examples.linear_wave.st_analysis.analysis import OUT_FILE as IN_FILE2

# position bubble labels
TEXT_POSITION_X = [0.12, 0.18, 0.023, 0.4, 0.27]
TEXT_POSITION_Y = [-0.9, -0.48, -0.54, -0.7, -1.5]
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

	for i in range(len(RS)):
		# define variables depending on R
		if i == 0:
			lims = [-0.01, 1, -2, 0]
			nums = [STOKES_HATS[0]] + STOKES_HATS[-2:]
			ylabel = r'$\bar{z}$'
		else:
			lims = [-0.1, 0.5, -1.5, 0]
			nums = STOKES_HATS[:3]
			ylabel = None

		# initialize subplot and create subplot labels
		fig(r'$\bar{u}/\epsilon^2$', ylabel, 121 + i, make_square=True,
			add_subplot_labels=True, width='jfm', lims=lims)

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
					 f'{sthat_gamma.iloc[j]:.2g}', bbox=PROPERTIES, fontsize=FS)
	plt.show()

if __name__ == '__main__':
	main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data
from examples.water_wave.st_analysis.numerics import STOKES_NUMS, BETAS, \
													 AMPLITUDE, WAVELENGTH
from examples.water_wave.st_analysis.numerics import OUT_FILE as IN_FILE1
from examples.water_wave.st_analysis.analysis import OUT_FILE as IN_FILE2

def main():
	"""
	Plot the drift velocity vs vertical position of particles in a wave.

	The numerical horizontal drift velocity and vertical position are averaged
	over each wave period. Solutions with and without history effects are
	included.
	"""
	numerics = pd.read_csv(IN_FILE1)
	analysis = pd.read_csv(IN_FILE2)
	k = 2 * np.pi / WAVELENGTH

	# position bubble labels
	text_position_x = [0.12, 0.18, 0.03, 0.4, 0.389]
	text_position_y = [-0.9, -0.5, -0.23, -0.9, -1.5]
	properties = dict(boxstyle='circle', fc='w', ec='k')

	for i in range(len(BETAS)):
		# define variables depending on beta
		ylabel = r'$\bar{z}$'
		if i == 0:
			lims = [-0.01, 1, -4, 0]
			nums = [STOKES_NUMS[0]] + STOKES_NUMS[-2:]
		else:
			lims = [-0.1, 1, -2, 0]
			nums = STOKES_NUMS[:3]
			ylabel = None

		# initialize subplot and create subplot labels
		fig(r'$\bar{u}$', ylabel, 121 + i, lims=lims, make_square=True,
			add_subplot_labels=True, width='jfm')

		# plot curves and data points
		for stokes_num, history in product(nums, [False, True]):
			# plot numerical solutions
			params = {'St': stokes_num, 'beta': BETAS[i], 'history': history}
			z_bar, u_bar = extract_data(['z', 'u_bar'], numerics, params)
			u_bar /= k * AMPLITUDE
			plt.scatter(u_bar, z_bar, marker='.', ec='k', fc='none')

			# plot fitted curves
			ls = '--' if history else '-'
			z_bar, u_bar = extract_data(['z_bar', 'u_bar'], analysis, params)
			plt.plot(u_bar, z_bar, c='k', ls=ls)

			# plot labels
			j = STOKES_NUMS.index(stokes_num)
			plt.text(text_position_x[j], text_position_y[j],
					 f'{stokes_num:.2f}', bbox=properties, fontsize=8)
	plt.show()

if __name__ == '__main__':
	main()

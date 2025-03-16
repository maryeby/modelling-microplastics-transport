import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

from utils.plot import initialize_figure as fig
from utils.plot import FS
from utils.data_tools import extract_data
from examples.water_wave.st_analysis.negatively_buoyant.numerics \
	 import STOKES_NUMS, AMPLITUDE, WAVELENGTH

IN_FILE1 = '../../../data/water_wave/st_numerics.csv'
IN_FILE2 = '../../../data/water_wave/st_analysis.csv'

def main():
	"""
	Plot the drift velocity vs vertical position of particles in a wave.

	The numerical horizontal drift velocity and vertical position are averaged
	over each wave period. Solutions with and without history effects are
	included.
	"""
	numerics = pd.read_csv(IN_FILE1)
	analysis = pd.read_csv(IN_FILE2)

	fig(r'$\bar{u}$', r'$\bar{z}$', lims=[0, 1, -4, 0.1], make_square=True)
	k = 2 * np.pi / WAVELENGTH

	# position bubble labels
	text_position_x = [0.5, 0.43, 0.31]
	text_position_y = [-0.5, -1.46, -2.85]
	properties = dict(boxstyle='circle', facecolor='w', edgecolor='k')

	for stokes_num, history in itertools.product(STOKES_NUMS, [False, True]):
		# plot numerical solutions
		params = {'St': stokes_num, 'history': history}
		z_bar, u_bar = extract_data(['z', 'u_bar'], numerics, params)
		u_bar /= k * AMPLITUDE
		plt.scatter(u_bar, z_bar, marker='.', edgecolors='k', facecolors='none',
					label='')

		# plot fitted curves
		label = 'with history effects' if history else 'without history effects'
		ls = '--' if history else '-'
		if stokes_num != STOKES_NUMS[-1]: label = ''
		z_bar, u_bar = extract_data(['z_bar', 'u_bar'], analysis, params)
		plt.plot(u_bar, z_bar, c='k', ls=ls, label=label)

		# plot labels
		i = STOKES_NUMS.index(stokes_num)
		plt.text(text_position_x[i], text_position_y[i], f'{stokes_num:.2f}',
				 fontsize=FS, bbox=properties)
	plt.legend(fontsize=FS)
	plt.show()

if __name__ == '__main__':
	main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from fractions import Fraction

from utils.data_tools import extract_data
from utils.plot import initialize_figure as fig
from utils.plot import FS
from utils.colors import COLORS
from examples.water_wave.beta_analysis.numerics import AMPLITUDE, WAVELENGTH, \
													   BETAS, R

IN_FILE1 = '../../data/water_wave/beta_numerics.csv'
IN_FILE2 = '../../data/water_wave/beta_analysis.csv'

def main():
	"""Plot the drift velocities of particles of varying densities in a wave."""
	numerics = pd.read_csv(IN_FILE1)
	analysis = pd.read_csv(IN_FILE2)
	fig(r'$\bar{u}$', r'$\bar{z}$', lims=[0, 1, -4, 0.1], make_square=True)

	# position bubble labels
	text_position_x = [0.6, 0.3, 0.4]
	text_position_y = [-0.3, -1.75, -0.76]
	properties = dict(boxstyle='circle', facecolor='w', edgecolor='k')

	for beta, history in itertools.product(BETAS, [True, False]):
		# plot analytical solution or fitted curve
		if beta == 1:
			z, u = extract_data(['z', 'u'], analysis, {'analytical': True})
			lc, ls = COLORS[3], '-'
			label = 'analytical solution' if history else ''
		else:
			params = {'beta': beta, 'history': history, 'analytical': False}
			z, u = extract_data(['z', 'u'], analysis, params)
			lc = 'k'
			label = 'with history effects' if history else \
					'without history effects'
			if beta != BETAS[-1]: label = ''
		plt.plot(u, z, c=lc, ls=ls, label=label, zorder=0)

		# plot numerical solutions
		params = {'beta': beta, 'history': history}
		ls = '--' if history else '-'
		z, u = extract_data(['z_crossings', 'u_bar'], numerics, params)
		k = 2 * np.pi / WAVELENGTH
		u /= k * AMPLITUDE
		plt.scatter(u, z, marker='.', edgecolors='k', facecolors='none',
					label='')
		# plot labels
		i = BETAS.index(beta)
		plt.text(text_position_x[i], text_position_y[i],
				 str(Fraction(R[i]).limit_denominator()), fontsize=FS,
				 bbox=properties)
	plt.legend(fontsize=FS)
	plt.show()

if __name__ == '__main__':
	main()

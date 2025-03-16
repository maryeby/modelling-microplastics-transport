import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from fractions import Fraction

from utils.plot import initialize_subplot as subplot
from utils.plot import FS
from utils.colors import COLORS
from utils.data_tools import extract_data
from examples.water_wave.st_analysis.negatively_buoyant.numerics \
	 import STOKES_NUMS
from examples.water_wave.st_analysis.negatively_buoyant.numerics \
	 import AMPLITUDE as ST_AMPLITUDE
from examples.water_wave.st_analysis.negatively_buoyant.numerics \
	 import WAVELENGTH as ST_WAVELENGTH
from examples.water_wave.beta_analysis.numerics import BETAS, R
from examples.water_wave.beta_analysis.numerics import AMPLITUDE as R_AMPLITUDE
from examples.water_wave.beta_analysis.numerics import WAVELENGTH \
	 as R_WAVELENGTH

IN_FILE1 = '../data/water_wave/st_numerics.csv'
IN_FILE2 = '../data/water_wave/st_analysis.csv'
IN_FILE3 = '../data/water_wave/beta_numerics.csv'
IN_FILE4 = '../data/water_wave/beta_analysis.csv'

def main():
	"""Plot the drift velocities of particles of varying sizes and densities."""
	# read data
	st_numerics = pd.read_csv(IN_FILE1)
	st_analysis = pd.read_csv(IN_FILE2)
	beta_numerics = pd.read_csv(IN_FILE3)
	beta_analysis = pd.read_csv(IN_FILE4)

	# initialize St analysis subplot
	plt.figure()
	subplot(121, r'$\bar{u}$', r'$\bar{z}$', lims=[0, 1, -4, 0.1],
			make_square=True)
	plt.gcf().text(0.02, 0.9, '(a)', fontsize=FS, fontfamily='serif',
				   fontstyle='italic')

	# position bubble labels
	text_position_x = [0.5, 0.43, 0.31]
	text_position_y = [-0.5, -1.46, -2.85]
	properties = dict(boxstyle='circle', facecolor='w', edgecolor='k')

	for stokes_num, history in product(STOKES_NUMS, [False, True]):
		# plot numerical solutions
		params = {'St': stokes_num, 'history': history}
		z_bar, u_bar = extract_data(['z', 'u_bar'], st_numerics, params)
		u_bar /= (2 * np.pi / ST_WAVELENGTH) * ST_AMPLITUDE
		plt.scatter(u_bar, z_bar, marker='.', edgecolors='k', facecolors='none',
					label='')

		# plot fitted curves
		label = 'with history effects' if history else 'without history effects'
		ls = '--' if history else '-'
		if stokes_num != STOKES_NUMS[-1]: label = ''
		z_bar, u_bar = extract_data(['z_bar', 'u_bar'], st_analysis, params)
		plt.plot(u_bar, z_bar, c='k', ls=ls, label=label)

		# plot labels
		i = STOKES_NUMS.index(stokes_num)
		plt.text(text_position_x[i], text_position_y[i], f'{stokes_num:.2f}',
				 fontsize=FS, bbox=properties)

	# position bubble labels
	text_position_x = [0.6, 0.3, 0.4]
	text_position_y = [-0.3, -1.75, -0.76]

	# initialize R analysis subplot
	subplot(122, r'$\bar{u}$', lims=[0, 1, -4, 0.1], make_square=True)
	plt.gcf().text(0.525, 0.9, '(b)', fontsize=FS, fontfamily='serif',
				   fontstyle='italic')

	for beta, history in product(BETAS, [False, True]):
		# plot analytical solution or fitted curve
		if beta == 1:
			z, u = extract_data(['z', 'u'], beta_analysis, {'analytical': True})
			lc, ls = COLORS[3], '-'
			label = 'analytical solution' if history else ''
		else:
			params = {'beta': beta, 'history': history, 'analytical': False}
			z, u = extract_data(['z', 'u'], beta_analysis, params)
			lc = 'k'
			label = 'with history effects' if history else \
					'without history effects'
			if beta != BETAS[-1]: label = ''
		plt.plot(u, z, c=lc, ls=ls, label=label, zorder=0)

		# plot numerical solutions
		params = {'beta': beta, 'history': history}
		ls = '--' if history else '-'
		z, u = extract_data(['z_crossings', 'u_bar'], beta_numerics, params)
		u /= (2 * np.pi / R_WAVELENGTH) * R_AMPLITUDE
		plt.scatter(u, z, marker='.', edgecolors='k', facecolors='none',
					label='', zorder=1)

		# plot labels
		i = BETAS.index(beta)
		plt.text(text_position_x[i], text_position_y[i],
				 str(Fraction(R[i]).limit_denominator()), fontsize=FS,
				 bbox=properties)
	plt.show()

if __name__ == '__main__':
	main()

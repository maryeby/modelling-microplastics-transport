import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from utils.plot import initialize_figure as fig
from utils.plot import FS
from utils.data_tools import extract_data
from examples.water_wave.critical_nums.analysis import DEPTH

SCALE = 2 / 3
DATA_PATH = '../../data/water_wave/critical_nums.csv'

def main():
	"""Plot the critical density ratio vs the critical Stokes number."""
	analysis = pd.read_csv(DATA_PATH) # read data

	# get relevant data
	fig(r'$R_c$', r'$St_c$', y_scale='log', make_square=True)
#	lims=[0, 2 / 3, 8e-3, 1]

	for history in [False, True]:
		beta_c, stokes_c = extract_data(['beta_c', 'St_c'], analysis,
										{'history': history})
		r_c = np.array(beta_c) * SCALE
		fmt = ':k.' if history else '-k.'
		lb = 'with history effects' if history else 'without history effects'
		plt.plot(r_c, stokes_c, fmt, label=lb)

#	non_monotonic_St = St_c_history[79]
#	non_monotonic_beta = beta_c_history[79]
#	plt.scatter(non_monotonic_beta, non_monotonic_St, c='hotpink')
#	plt.annotate(r'$\beta_c =$' + f'{non_monotonic_beta:.4f}',
#				 (non_monotonic_beta - 0.2, non_monotonic_St), fontsize=fs,
#				 ha='left')
#	plt.legend(fontsize=fs)

	# plot critical beta vs final z_crossings
	fig(r'$R_c$', r'$z-crossing_f$')
	for history in [False, True]:
		beta_c, z_f = extract_data(['beta_c', 'z_crossing_f'], analysis,
								   {'history': history})
		r_c = np.array(beta_c) * SCALE
		m = 'o' if history else 's'
		lb = 'with history effects' if history else 'without history effects'
		plt.scatter(r_c, z_f, marker=m, facecolor='none', edgecolor='k',
					label=lb)
	plt.axhline(-2 * np.pi * DEPTH, c='silver', ls=':')
	plt.legend(fontsize=FS)
	plt.show()

if __name__ == '__main__':
	main()

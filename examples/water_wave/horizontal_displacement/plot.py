import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data
from examples.water_wave.horizontal_displacement.numerics import DB_ST, DB_BETA
from examples.water_wave.horizontal_displacement.analysis import OUT_FILE as \
	 IN_FILE

NUM_BINS = 200

def main():
	r"""Plot *St* and $\beta$ vs the % change in horizontal displacement."""
	# plot St vs % horizontal displacement
	analysis = pd.read_csv(IN_FILE)
#	fig(r'$St$', r'$\%\Delta x_f$', 121, make_square=True,
#		add_subplot_labels=True)
	names = ['St', 'mean', 'max', 'min']
	stokes_nums, x, max_val, min_val = extract_data(names, analysis,
													{'beta': DB_BETA})
	max_val, min_val = np.abs(max_val), np.abs(min_val)
#	plt.errorbar(stokes_nums, x, [max_val, min_val], fmt='-k.', capsize=5)

	# plot beta vs % horizontal displacement
#	fig(r'$\beta$', num=122, make_square=True)
	fig(r'$\beta$', r'$\%\Delta x_f$', make_square=True)
	names[0] = 'beta'
	beta, x, max_val, min_val = extract_data(names, analysis,
											{'St': DB_ST})
	max_val, min_val = np.abs(max_val), np.abs(min_val)
	plt.errorbar(beta, x, [max_val, min_val], fmt='-k.', capsize=5)

	# plot histogram of % horizontal displacement data
	fig(r'%$\Delta x_f$')
	plt.hist('percent_displacement', bins=NUM_BINS, data=analysis, color='gray')
	plt.show()

if __name__ == '__main__': main()

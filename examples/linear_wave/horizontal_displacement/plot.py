import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data
from examples.linear_wave.horizontal_displacement.dibenedetto_numerics import \
	 AMPLITUDE, WAVELENGTH
from examples.linear_wave.horizontal_displacement.numerics import DB_ST, DB_R
from examples.linear_wave.horizontal_displacement.analysis import OUT_FILE as \
	 IN_FILE

NUM_BINS = 200

def main():
	r"""Plot *R* vs the change in horizontal displacement."""
	analysis = pd.read_csv(IN_FILE)
	names = ['R', 'mean', 'max', 'min']

	# plot R vs % horizontal displacement
	fig(r'$R$', r'$\Delta x$', make_square=False, width='jfm')
	r, x, max_val, min_val = extract_data(names, analysis, {'Sthat': DB_ST})
	max_val, min_val = np.abs(max_val), np.abs(min_val)
	plt.errorbar(r, np.array(x), [max_val, min_val], fmt='-k.', capsize=5)

	# plot histogram of % horizontal displacement data
	fig(r'%$\Delta x_f$')
	plt.hist('percent_displacement', bins=NUM_BINS, data=analysis, color='gray')
	plt.show()

if __name__ == '__main__': main()

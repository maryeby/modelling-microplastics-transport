import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data, print_parameter
from examples.linear_wave.horizontal_displacement.dibenedetto_numerics import \
	 AMPLITUDE, WAVELENGTH
from examples.linear_wave.horizontal_displacement.numerics import DB_ST, DB_R
from examples.linear_wave.horizontal_displacement.numerics import OUT_FILE as \
	 IN_FILE1
from examples.linear_wave.horizontal_displacement.analysis import OUT_FILE as \
	 IN_FILE2

YMIN, YMAX = 0, 0.45
NUM_BINS = 200

def main():
	r"""Plot *R* vs the change in horizontal displacement."""
	numerics = pd.read_csv(IN_FILE1)
	analysis = pd.read_csv(IN_FILE2)
	names = ['Sthat', 'mean', 'max', 'min']
	print_parameter('epsilon', 2 * np.pi / WAVELENGTH * AMPLITUDE)

	# plot St vs horizontal displacement
	fig(r'$\widehat{St}$', r'$\Delta x$', 121,
		add_subplot_labels=True)
	st, x, max_val, min_val = extract_data(names, analysis, {'R': DB_R})
	max_val, min_val, x = np.abs(max_val), np.abs(min_val), x.to_numpy()
	plt.ylim(YMIN, YMAX)
	plt.fill_between(st, min_val, max_val, color='silver')
	plt.plot(st, x, '-k.')

	# plot R vs horizontal displacement
	names[0] = 'R'
	fig(r'$R$', num=122, hide_yticks=True, add_subplot_labels=True)
	r, x, max_val, min_val = extract_data(names, analysis, {'Sthat': DB_ST})
	max_val, min_val, x = np.abs(max_val), np.abs(min_val), x.to_numpy()
	plt.ylim(YMIN, YMAX)
	plt.fill_between(r, min_val, max_val, color='silver')
	plt.plot(r, x, '-k.')

	# plot histogram of % horizontal displacement data
	fig(r'$\Delta x_f$')
	plt.hist('delta_x', bins=NUM_BINS, data=analysis, color='gray')

	# find particle trajectory with the largest difference in displacement
	max_displacement = np.max(analysis['delta_x'])
	params = {'delta_x': max_displacement}
	max_params = extract_data(['Sthat', 'R', 'x_0'], analysis, params)
	sthat, r, x_0 = [p.iloc[0] for p in max_params]
	params = {'Sthat': sthat, 'R': r, 'x_0': x_0, 'history': False}
	x, z = extract_data(['x', 'z'], numerics, params)

	# plot particle trajectory
	fig(r'$x$', r'$z$', make_square=True, equal_aspect=False)
	plt.plot(x, z, '-k', label='without history effects')
	params['history'] = True
	x, z = extract_data(['x', 'z'], numerics, params)
	plt.plot(x, z, '--k', label='with history effects')
	plt.show()

if __name__ == '__main__': main()

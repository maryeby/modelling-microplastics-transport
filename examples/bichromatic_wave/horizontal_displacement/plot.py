import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data, print_parameter
from examples.linear_wave.horizontal_displacement.numerics import DB_ST, DB_R
from examples.bichromatic_wave.trajectories.numerics import AMPLITUDES, \
															WAVELENGTHS, SLOPE
from examples.bichromatic_wave.horizontal_displacement.numerics import OUT_FILE\
	 as IN_FILE1
from examples.bichromatic_wave.horizontal_displacement.analysis import OUT_FILE\
	 as IN_FILE2

YMIN, YMAX = [-0.5, -1], [3, 15]
SHOW_TITLES = False

def main():
	r"""Plot $\widehat{St}$ and $R$ vs the change in horizontal displacement."""
	numerics = pd.read_csv(IN_FILE1)
	analysis = pd.read_csv(IN_FILE2)
	print_parameter('epsilon', 2 * np.pi / WAVELENGTHS[0] * AMPLITUDES[0])

	for m, ymin, ymax in zip([0, SLOPE], YMIN, YMAX):
		names = ['Sthat', 'mean', 'max', 'min']
		title = 'flat seabed' if m == 0 else 'sloped seabed'
		params = {'R': DB_R, 'slope': m}

		# plot St vs horizontal displacement
		fig(r'$\widehat{St}$', r'$\Delta x$', 121, add_subplot_labels=True)
		if SHOW_TITLES: plt.suptitle(title)
		st, x, max_val, min_val = extract_data(names, analysis, params)
		max_val, min_val, x = np.abs(max_val), np.abs(min_val), x.to_numpy()
		plt.ylim(ymin, ymax)
		plt.fill_between(st, min_val, max_val, color='silver')
		plt.plot(st, x, '-k.')

		# plot R vs horizontal displacement
		names[0] = 'R'
		params = {'Sthat': DB_ST, 'slope': m}
		fig(r'$R$', num=122, hide_yticks=True, add_subplot_labels=True)
		r, x, max_val, min_val = extract_data(names, analysis, params)
		max_val, min_val, x = np.abs(max_val), np.abs(min_val), x.to_numpy()
		plt.ylim(ymin, ymax)
		plt.fill_between(r, min_val, max_val, color='silver')
		plt.plot(r, x, '-k.')

		# find particle trajectory with the largest difference in displacement
		max_displacement = np.max(analysis['delta_x'])
		params = {'delta_x': max_displacement}
		max_params = extract_data(['Sthat', 'R', 'x_0'], analysis, params)
		sthat, r, x_0 = [p.iloc[0] for p in max_params]
		params = {'Sthat': sthat, 'R': r, 'x_0': x_0, 'history': False,
				  'slope': m}
		x, z = extract_data(['x', 'z'], numerics, params)

		# plot particle trajectory
		fig(r'$x$', r'$z$', make_square=True, equal_aspect=False)
		plt.title(title + rf', $\widehat St = ${sthat:.5g}, $R = ${r:.5g},'
						+ rf' $x_0 = {x_0:.5g}$')
		plt.plot(x, z, '-k', label='without history effects')
		params['history'] = True
		x, z = extract_data(['x', 'z'], numerics, params)
		plt.plot(x, z, '--k', label='with history effects')
	plt.show()

if __name__ == '__main__': main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data, print_parameter
from examples.linear_wave.horizontal_displacement.dibenedetto_numerics import \
	 DEPTH, WAVELENGTH
from examples.linear_wave.horizontal_displacement.numerics import DB_ST, DB_R
from examples.stokes_wave.horizontal_displacement.numerics import AMPLITUDE
from examples.stokes_wave.horizontal_displacement.numerics import OUT_FILE as \
	 IN_FILE1
from examples.linear_wave.horizontal_displacement.numerics import OUT_FILE as \
	 IN_FILE2
from examples.stokes_wave.horizontal_displacement.history_analysis \
	 import OUT_FILE as IN_FILE3
from examples.stokes_wave.horizontal_displacement.linear_analysis \
	 import OUT_FILE as IN_FILE4

ALPHA = 0.8 # transparency of shading

def main():
	r"""Plot $R$ and $\hat{St}$ vs the change in horizontal displacement."""
	names = ['Sthat', 'mean', 'max', 'min']
	print_parameter('Sthat', DB_ST)
	print_parameter('R', DB_R)
	print_parameter('epsilon', 2 * np.pi / WAVELENGTH * AMPLITUDE)

	# variables for plotting displacement between particles with and w/o history
	numerics = pd.read_csv(IN_FILE1)
	analysis = pd.read_csv(IN_FILE3)
	ymin, ymax = -0.02, 0.45

	# plot St vs horizontal displacement
	fig(r'$\widehat{St}$', r'$\Delta x$', 121, add_subplot_labels=True)
	st, x, max_val, min_val = extract_data(names, analysis, {'R': DB_R})
	max_val, min_val, x = np.abs(max_val), np.abs(min_val), x.to_numpy()
	plt.fill_between(st, min_val, max_val, color='silver')
	plt.ylim(ymin, ymax)
	plt.plot(st, x, '-k.')

	# plot R vs horizontal displacement
	names[0] = 'R'
	fig(r'$R$', num=122, hide_yticks=True, add_subplot_labels=True)
	r, x, max_val, min_val = extract_data(names, analysis, {'Sthat': DB_ST})
	max_val, min_val, x = np.abs(max_val), np.abs(min_val), x.to_numpy()
	plt.fill_between(r, min_val, max_val, color='silver')
	plt.ylim(ymin, ymax)
	plt.plot(r, x, '-k.')

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
	plt.legend()

	# variables for plotting displacement between linear/non-linear waves
	ymin, ymax = 0, 64
	analysis = pd.read_csv(IN_FILE4)
	names[0] = 'Sthat'

	# plot St vs horizontal displacement for linear vs nonlinear waves
	fig(r'$\widehat{St}$', r'$\Delta x$', 121, add_subplot_labels=True)
	for h in [False, True]:
		shade = 'silver' if h else 'grey'
		fmt = '--k.' if h else '-k.'
		params = {'R': DB_R, 'history': h}
		st, x, max_val, min_val = extract_data(names, analysis, params)
		max_val, min_val, x = np.abs(max_val), np.abs(min_val), x.to_numpy()
		plt.fill_between(st, min_val, max_val, color=shade, alpha=ALPHA)
		plt.ylim(ymin, ymax)
		plt.plot(st, x, fmt)

	# plot R vs horizontal displacement for linear vs nonlinear waves
	names[0] = 'R'
	fig(r'$R$', num=122, hide_yticks=True, add_subplot_labels=True)
	for h in [False, True]:
		shade = 'silver' if h else 'grey'
		fmt = '--k.' if h else '-k.'
		params = {'Sthat': DB_ST, 'history': h}
		r, x, max_val, min_val = extract_data(names, analysis, params)
		max_val, min_val, x = np.abs(max_val), np.abs(min_val), x.to_numpy()
		plt.fill_between(r, min_val, max_val, color=shade, alpha=ALPHA)
		plt.ylim(ymin, ymax)
		plt.plot(r, x, fmt)

	# find particle trajectory with the largest difference in displacement
	names = ['Sthat', 'R', 'x_0', 'history']
	max_displacement = np.max(analysis['delta_x'])
	params = {'delta_x': max_displacement}
	max_params = extract_data(names, analysis, params)
	sthat, r, x_0, history = [p.iloc[0] for p in max_params]
	params = {'Sthat': sthat, 'R': r, 'x_0': x_0, 'history': history}
	x, z = extract_data(['x', 'z'], numerics, params)

	# plot particle trajectory
	fig(r'$x$', r'$z$', make_square=True, equal_aspect=False)
	plt.plot(x, z, '--k', label='non-linear wave')
	numerics = pd.read_csv(IN_FILE2)
	x, z = extract_data(['x', 'z'], numerics, params)
	plt.plot(x, z, '-k', label='linear wave')
	plt.legend()
	plt.show()

if __name__ == '__main__': main()

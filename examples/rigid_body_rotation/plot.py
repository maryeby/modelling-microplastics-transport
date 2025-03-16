import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from utils.plot import initialize_subplot as subplot
from utils.plot import FS
from utils.data_tools import extract_data

IN_FILE1 = '../data/rigid_body_rotation/numerics.csv'
IN_FILE2 = '../data/rigid_body_rotation/analytics.csv'
IN_FILE3 = '../data/rigid_body_rotation/history.csv'
IN_FILE4 = '../data/rigid_body_rotation/rel_error.csv'
IN_FILE5 = '../data/rigid_body_rotation/global_error.csv'
IN_FILE6 = '../data/rigid_body_rotation/daitche_fig3.csv'
STYLES = ['--', '-.', ':']
LABELS = ['first order', 'second order', 'third order']

def main():
	"""
	Plot solutions for a rotating rigid body, and corresponding error analysis.

	The plots produced include numerical solutions, analytical solutions,
	relative error analysis, and global error analysis to reproduce [1] figures
	3 and 4.
	
	References
	----------
	[^1]: [A. Daitche (2013).](https://doi.org/10.1016/j.jcp.2013.07.024)
		  Advection of inertial particles in the presence of the history force:
		  Higher order numerical schemes. *Journal of Computational Physics*
		  254, 93–106.
	"""
	# read data
	numerics = pd.read_csv(IN_FILE1)
	analytics = pd.read_csv(IN_FILE2)
	history = pd.read_csv(IN_FILE3)
	rel_error = pd.read_csv(IN_FILE4)
	global_error = pd.read_csv(IN_FILE5)
	daitche = pd.read_csv(IN_FILE6)

	# initialize trajectory figure
	fig(r'$x$', r'$z$', lims=[-2, 2.5, -2.5, 2], make_square=True,
												equal_aspect=True)
	# plot numerical trajectory
	for i in range(3):
		fmt = STYLES[i] + 'k'
		x, z = extract_data(['x', 'z'], numerics, {'order': i + 1})
		x, z = crop([x, z], 2000)
		plt.plot(x, z, fmt, label=LABELS[i])

	# plot analytical trajectory
	x, z = extract_data(['x', 'z'], analytics, {'delta_t': 1e-2})
	x, z = crop([x, z], 2000)
	plt.plot(x, z, c='k', label='exact')
	plt.legend(fontsize=FS, loc='center', frameon=False)

	# plot extracted first order data and integer time points
	plt.scatter('first_x', 'first_z', c='silver', marker='x', data=daitche)
	x, z = extract_data(['x', 'z'], analytics, {'delta_t': 1e-2})
	x, z = crop([x, z], -20, reverse=True)
	plt.scatter(x, z, c='k')

	# plot relative error (recreation of Daitche (2013) Figure 3(b))
	t = extract_data('t', rel_error, {'order': 3})
	fig(r'$t$', r'$E_{rel}$', lims=[0, 100, 1e-7, 1e0], y_scale='log')
	for i in range(3):
		fmt = STYLES[i] + 'k'
		e_rel = extract_data('e_rel', rel_error, {'order': i + 1})
		plt.plot('t' + str(i + 1), 'rel_error' + str(i + 1), c='silver',
				 data=daitche, label='')
		plt.plot(t, e_rel, fmt, label=LABELS[i])
	plt.legend(fontsize=FS)

	# plot global error (recreation of Daitche (2013) Figure 4)
	fig(r'$\Delta t$', r'$\mathcal{\epsilon}$', x_scale='log', y_scale='log',
		lims=[1e-3, 1e-1, 1e-11, 1e3])
	h_scale = np.linspace(2e-3, 5e-2, 10)
	h_labels = [r'~$h$', r'~$h^2$', '~$h^3$']
	for i in range(3):
		fmt = STYLES[i] + 'k.'
		e_global = extract_data('global_error', global_error, {'order': i + 1})
		delta_t = extract_data('delta_t', global_error, {'order': i + 1})
		plt.plot(h_scale, h_scale ** (i + 1), c='grey', ls=STYLES[i],
				 label=h_labels[i])
		plt.plot(delta_t, e_global, fmt, label=LABELS[i])
	plt.legend(fontsize=FS)

	# plot absolute error
	fig(r'$t$', r'$E_{abs}$', y_scale='log', lims=[-1, 10, 1e-8, 1e-5])
	for i in range(3):
		fmt = STYLES[i] + 'k'
		e_abs = extract_data('e_abs', rel_error, {'order': i + 1})
		plt.plot(t, e_rel, fmt, label=LABELS[i])
	plt.legend(fontsize=FS)

	# plot vertical history force values at t = 0
	fig(r'$\Delta t$', r'$H\'(0)_z$')
	delta_t, history_z = extract_data(['delta_t', 'history_z'], history,
									  {'t': 0})
	plt.plot(delta_t, history_z, '-k.')

	# extract analytical solutions for the history force
	delta_t = 1e-2
	names = ['t', 'history_x', 'history_z']
	params = {'delta_t': delta_t}
	t, exact_x, exact_z = extract_data(names, analytics, params)
	t, exact_x, exact_z = crop([t, exact_x, exact_z], 998)

	# extract numerical solutions for the history force
	del(names[0])
	history_x, history_z = extract_data(names, history, params)
	history_x, history_z = crop([history_x, history_z], -2)

	# plot analytical and numerical solutions for the history force
	plt.figure()
	subplot(211, y_label=r'$H\'(t)_x$')
	plt.plot(t, exact_x, c='silver')
	plt.plot(t, history_x, ':k')

	subplot(212, r'$t$', r'$H\'(t)_z$')
	plt.plot(t, exact_z, c='silver')
	plt.plot(t, history_z, ':k')

	# plot computation time
	fig(r'$\Delta t$', 'computation time (s)', x_scale='log', y_scale='log')
	for i in range(3):
		fmt = STYLES[i] + 'k.'
		delta_t, comp_time = extract_data(['delta_t', 'computation_time'],
										  global_error, {'order': i + 1})
		plt.plot(delta_t, comp_time, fmt, label=LABELS[i])
	plt.legend(fontsize=FS)
	plt.show()

def crop(lst, n, reverse=False):
	"""Slice each Series in `lst` using `n`."""
	if reverse:
		return [s.iloc[n:] for s in lst]
	else:
		return [s.iloc[:n] for s in lst]

if __name__ == '__main__':
	main()

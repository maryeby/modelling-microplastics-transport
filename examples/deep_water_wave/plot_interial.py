import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from utils.plot import initialize_figure as fig
from utils.plot import FS
from utils.data_tools import extract_data
from utils.colors import COLORS

IN_FILE1 = '../data/deep_water_wave/inertial_numerics.csv'
IN_FILE2 = '../data/deep_water_wave/global_error.csv'
STYLES = ['--', '-.', ':']
LABELS = ['first order', 'second order', 'third order']
WIDTHS = [3, 2, 1]

def main():
	"""
	Plot the results computed in `inertial_numerics.py`.

	Leading order, first order, and second order inertial equation solutions are
	included as derived in [1, 2], as well as numerical solutions generated
	using the multi-step integration method,[^3] and the associated global
	error.

	References
	----------
	[^1]: [F. Santamaria et al. (2013).](
		  https://doi.org/10.1209/0295-5075/102/14003)
		  Stokes drift for inertial particles transported by water waves.
		  *EPL (Europhysics Letters)*, 102(1), 14003.
	[^2]: [G. Haller & T. Sapsis (2008).](
		  https://doi.org/10.1016/j.physd.2007.09.027)
		  Where do inertial particles go in fluid flows?
		  *Physica D: Nonlinear Phenomena* 237(5), 573–583.
	[^3]: [A. Daitche (2013).](https://doi.org/10.1016/j.jcp.2013.07.024)
		  Advection of inertial particles in the presence of the history force:
		  Higher order numerical schemes. *Journal of Computational Physics*
		  254, 93–106.
	"""
	# read data
	numerics = pd.read_csv(IN_FILE1)
	global_error = pd.read_csv(IN_FILE2)
	methods = numerics['method'].drop_duplicates().tolist()

	fig('x', 'z', equal_aspect=True)
	for order, method in itertools.product(range(3), methods[:2]):
		# extract inertial equation solutions to plot
		i = methods.index(method)
		params = {'equation': 'inertial', 'order': order, 'method': method}
		x, z = extract_data(['x', 'z'], numerics, params)

		# define line color, width, style, and label, then plot
		lc, lw, ls = COLORS[i], WIDTHS[i], STYLES[order]
		label = f'order {order}' if i == 0 else ''
		plt.plot(x, z, c=lc, ls=ls, lw=lw, marker='o', label=label)

	for i in range(len(methods)):
		# extract M-R solutions to plot
		params = {'equation': 'Maxey-Riley', 'method': methods[i]}
		x, z = extract_data(['x', 'z'], numerics, params)

		# define line color, width, and label, then plot
		lc, lw = COLORS[i], WIDTHS[i]
		label = 'Maxey-Riley' if i == 0 else ''
		plt.plot(x, z, c=lc, lw=lw, marker='o', label=label)
	plt.legend(fontsize=FS)

	# create global error figure and plot reference lines (h, h^2, h^3)
	fig(r'$\Delta t$', r'$\mathcal{\epsilon}$', x_scale='log', y_scale='log',
		make_square=True)
	h_scale = np.linspace(2e-3, 2e-2, 10)
	plt.plot(h_scale, h_scale * 2, c=COLORS[-1], ls=STYLES[0], label=r'~$h$')
	plt.plot(h_scale, (h_scale ** 2) * 2.3, c=COLORS[-1], ls=STYLES[1],
			 label=r'~$h^2$')
	for order in [1, 2, 3]:
		params = {'order': order}
		delta_t, error = extract_data(['delta_t', 'global_error'], global_error,
									  params)
		fmt = STYLES[order - 1] + 'k.'
		plt.plot(delta_t, error, fmt, label=LABELS[order - 1])
	plt.legend(fontsize=FS)
	plt.show()

if __name__ == '__main__':
	main()

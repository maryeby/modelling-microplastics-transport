import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data
from examples.relaxing_particle.experimental_comparison.numerics import D_PS, \
	 U_T
from examples.relaxing_particle.experimental_comparison.numerics import \
	 OUT_FILE as IN_FILE1

NUS = np.array([0.0011, 1e-6])
T_O = NUS / U_T ** 2
PLOT_NUMS = [121, 122]
XLABEL, YLABELS = r"$t'/T'_O$", [r"$\Delta z'/(2a')$", r"$\dot{z}'/U'_T$"]
LIMITS = [i for i in [[5e-3, 1.1, 8e-3, 13], [5e-3, 1.1, 0.4, 1.1],
					  [None] * 4, [None] * 4] for j in
		  range(len(D_PS))]

IN_FILE2 = '../../data/relaxing_particle/jaroslawski_fig2a.csv'
IN_FILE3 = '../../data/relaxing_particle/jaroslawski_fig2b.csv'

def main():
	"""
	Plot the vertical particle displacement and velocity over time.

	The plots include a recreation of Figure 2 from Ref. 1, and an example
	created with parametric ranges relevant to microplastics.

	References
	----------
	[^1]: [T. Jaroslawski et al. (2025).](https://doi.org/10.1103/PhysRevFluids.
		  10.L062301) Stokesian settling from quiescence: Experiments and theory
		  on history effects and unsteady flow structures. *Physical Review
		  Fluids* 10(6), L062301-1–L062301-10.
	"""
	# read data
	numerics = pd.read_csv(IN_FILE1)
	fig2a = pd.read_csv(IN_FILE2)
	fig2b = pd.read_csv(IN_FILE3)

	names = ['t', 'z', 'zdot']
	for (dp, to), (num, ylabel) in product(zip(D_PS, T_O), zip(PLOT_NUMS,
															   YLABELS)):
		# format subplot
		yscale = 'log' if num == PLOT_NUMS[0] else None
		fig(XLABEL, ylabel, num, x_scale='log', y_scale=yscale,
			add_subplot_labels=True)

		for history in [False, True]:
			# extract data
			params = {'history': history, 'd_p': dp}
			t, z, zdot = extract_data(names, numerics, params)
			ydata = np.abs(z) / dp if num == PLOT_NUMS[0] else np.abs(zdot)

			# plot data
			fmt = '--k' if history else '-k'
			plt.plot(t / to, ydata, fmt)
			if dp == D_PS[0] and history:
				ta, y = extract_data(['t/T', 'y/dp'], fig2a)
				tb, v = extract_data(['t/T', 'v/U_T'], fig2b)
				exp_t = ta if num == PLOT_NUMS[0] else tb
				exp_ydata = y if num == PLOT_NUMS[0] else v
				plt.scatter(exp_t, exp_ydata, ec='k', fc='none')
		plt.xlim(right=1)
	plt.show()

if __name__ == '__main__':
	main()

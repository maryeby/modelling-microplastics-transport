import pandas as pd
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from utils.plot import FS
from utils.data_tools import extract_data
from utils.colors import COLORS
from examples.relaxing_particle.numerics import BETAS

IN_FILE1 = '../data/relaxing_particle/numerics.csv'
IN_FILE2 = '../data/relaxing_particle/asymptotics.csv'
IN_FILE3 = '../data/relaxing_particle/prasath_fig4.csv'

def main():
	"""
	Plot the horizontal velocity of a relaxing particle over time.

	The plot includes numerical and asymptotic results for a particle in a
	quiescent flow, reproducing the results from [1] Figure 4.

	References
	----------
	[^1]: [S. G. Prasath et al. (2019)](https://doi.org/10.1017/jfm.2019.194)
		  Accurate solution method for the Maxey–Riley equation, and the
		  effects of Basset history. *Journal of Fluid Mechanics* 868, 428–460.
	"""
	# read data and create figure
	numerics = pd.read_csv(IN_FILE1)
	asymptotics = pd.read_csv(IN_FILE2)
	prasath = pd.read_csv(IN_FILE3)
	fig(r'$t$', r'$\dot{x}$', lims=[0, 14.5, 1e-5, 1e1], y_scale='log')

	# plot
	for i in range(len(BETAS)):
		# plot asymptotic solution
		params = {'beta': BETAS[i]}
		t, xdot = extract_data(['t', 'xdot'], asymptotics, params)
		plt.plot(t, xdot, c=COLORS[i], ls=':')

		# plot numerical solution without history
		params['history'] = False
		t, xdot = extract_data(['t', 'xdot'], numerics, params)
		plt.plot(t, xdot, c=COLORS[i], ls='--')

		# plot numerical solution with history
		params['history'] = True
		xdot = extract_data('xdot', numerics, params)
		plt.plot(t, xdot, c=COLORS[i], label=r'$\beta =$' + f'{BETAS[i]:g}')

		# plot asymptotic solution extracted from Prasath et al. (2019) Fig. 4
		params['asymptotic'] = True
		t, xdot = extract_data(['t', 'xdot'], prasath, params)
		plt.plot(t, xdot, c=COLORS[-1], ls=':')

		# plot numerical sol with history extracted from Prasath Fig. 4
		params['asymptotic'] = False
		t, xdot = extract_data(['t', 'xdot'], prasath, params)
		plt.plot(t, xdot, c=COLORS[-1])

		# plot numerical sol without history extracted from Prasath Fig. 4
		params['history'] = False
		t, xdot = extract_data(['t', 'xdot'], prasath, params)
		plt.plot(t, xdot, c=COLORS[-1], ls='--')
	plt.legend(fontsize=FS)
	plt.show()

if __name__ == '__main__':
	main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.data_tools import extract_data
from utils.plot import initialize_figure as fig
from examples.water_wave.forces.compute_st_star import BETAS
from examples.water_wave.forces.compute_st_star import OUT_FILE as IN_FILE

def main():
	r"""Plot $St^*$ as a function of the wave steepness $\epsilon$."""
	analysis = pd.read_csv(IN_FILE)
	fig(r'$\epsilon$', r'$St^*$', make_square=True)

	names = ['epsilon', 'St*', 'R']
#	for beta, ls in zip(BETAS, ['--', '-', ':']): # plot all curves
	for beta, ls in zip([BETAS[1]], ['-']): # plot only neutrally buoyant curve
		epsilon, st_star, r = extract_data(names, analysis, {'beta': beta})
		r = r.iloc[0] if r.drop_duplicates().size == 1 else None
		fmt = ls + 'k.'
		plt.plot(epsilon, st_star, fmt, label=f'R = {r:.2f}')
		if beta == BETAS[1]: # shade under curve
			plt.fill_between(epsilon, st_star, st_star.iloc[0],
							 color='lightgrey')
	plt.text(0.06, 0.0195, 'history force dominates',
			 horizontalalignment='center', verticalalignment='center')
	plt.text(0.08, 0.0125, 'Stokes drag dominates',
			 horizontalalignment='center', verticalalignment='center')
#	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()

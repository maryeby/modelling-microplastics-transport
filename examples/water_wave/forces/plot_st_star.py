import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.data_tools import extract_data
from utils.plot import initialize_figure as fig
from utils.colors import COLORS
from examples.water_wave.forces.compute_st_star import BETAS, RS, EPSILONS
from examples.water_wave.forces.compute_st_star import OUT_FILE as IN_FILE

def main():
	r"""Plot $St^*$ as a function of the wave steepness $\epsilon$."""
	analysis = pd.read_csv(IN_FILE)
	fig(r'$\epsilon$', r'$St^*$', make_square=True, width='jfm')
	print('\nSthat*\tSthat75\t  R')

	for beta, r, color in zip(BETAS, RS, COLORS):
		# plot 100% lines
		names = ['epsilon', 'St*', 'Sthat*']
		param = {'beta': beta}
		epsilon, st_star, st_hat_star = extract_data(names, analysis, param)
		plt.plot(epsilon, st_star, c=color, marker='.', label=f'R = {r:.4f}')
		print(f'{st_hat_star.mean():.4f}', end='')
		if beta == BETAS[4]: plt.fill_between(epsilon, st_star, st_star.iloc[0],
							 color='lightgrey') # shading under the curve
		# plot 75% lines
		names = ['epsilon', 'St75', 'Sthat75']
		epsilon, st75, st_hat_75 = extract_data(names, analysis, param)
		plt.plot(epsilon, st75, c=color, ls='--', marker='x', label='')
		print(f'\t{st_hat_75.mean():.4f}\t{r:.4f}')

#	plt.text(0.06, 0.03, 'history force dominates',
#			 horizontalalignment='center', verticalalignment='center')
#	plt.text(0.08, 0.014, 'Stokes drag dominates',
#			 horizontalalignment='center', verticalalignment='center')
	print()
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()

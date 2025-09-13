import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data
from examples.water_wave.critical_nums.trajectory_numerics import BETAS
from examples.water_wave.critical_nums.analysis import SCALE

TITLES = ['pre-critical', 'critical', 'post-critical']
IN_FILE = '../../data/water_wave/critical_trajectories.csv'

def main():
	"""Plot the trajectories of particles of critical size and density."""
	numerics = pd.read_csv(IN_FILE)
	stokes_nums = numerics['St'].drop_duplicates().to_numpy()

	# plot trajectories to look at their topology
	for beta, stokes_num in zip(BETAS[:-3], stokes_nums[:-3]):
		fig(r'$x$', r'$z$', make_square=True, width='jfm')
		beta_str = r'$\beta = $'
		title = rf'$R$ = {beta * SCALE:.4f} ({beta_str}{beta:g}), ' \
			  + rf'$St$ = {stokes_num:.4f}'
		plt.title(title)
		x, z = extract_data(['x', 'z'], numerics, {'beta': beta})
		xc, zc = extract_data(['x_crossings', 'z_crossings'], numerics,
							  {'beta': beta})
		plt.plot(x, z, '-k')
		plt.scatter(xc, zc, edgecolors='k', facecolors='none')

	# plot pre-critical, critical, post-critical trajectories
	for i, j in zip(range(len(TITLES)), [-3, -2, -1]):
		fig(r'$x$', num=131 + i, make_square=False, width='jfm',
			lims=[-0.01, 0.11, -7, 0.15])
		if i == 0 and j == -3:
			plt.ylabel(r'$z$')
			plt.yticks([0, -2, -4, -6])
#		plt.title(TITLES[i])
		plt.xticks([0, 0.05, 0.1])
		plt.yticks([0, -5])
		x, z = extract_data(['x', 'z'], numerics, {'beta': BETAS[j],
												   'St': stokes_nums[j]})
		plt.plot(x, z, '-k')
	plt.show()
if __name__ == '__main__': main()

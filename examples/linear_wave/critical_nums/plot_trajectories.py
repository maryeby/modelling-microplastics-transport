import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data
from examples.linear_wave.critical_nums.trajectory_numerics import R
from examples.linear_wave.critical_nums.trajectory_numerics import OUT_FILE as \
																   IN_FILE

TITLES = ['pre-critical', 'critical', 'post-critical']

def main():
	"""Plot the trajectories of particles of critical size and density."""
	numerics = pd.read_csv(IN_FILE)
	stokes_nums = numerics['St'].drop_duplicates().to_numpy()

	# plot pre-critical, critical, post-critical trajectories
	for i, j in zip(range(len(TITLES)), [-3, -2, -1]):
		if i == 0:
			l = [-0.04, 0.11, -2.3, 0.1]
			ytix = [0, -1, -2]
		else:
			l = [-0.01, 0.11, -7, 0.15]
			ytix = [0, -2, -4, -6]
		fig(r'$x$', num=131 + i, make_square=False, lims=l)
		if i == 0 and j == -3: plt.ylabel(r'$z$')
#		plt.title(TITLES[i])
		plt.xticks([0, 0.05, 0.1])
		plt.yticks(ytix)
		x, z = extract_data(['x', 'z'], numerics, {'St': stokes_nums[j]})
		plt.plot(x, z, '-k')
	plt.show()
if __name__ == '__main__': main()

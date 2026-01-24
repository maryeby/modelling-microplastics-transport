import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from fractions import Fraction

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data
from examples.stokes_wave.critical_nums.trajectory_numerics import R
from examples.stokes_wave.critical_nums.trajectory_numerics import OUT_FILE as \
																   IN_FILE

TITLES = ['pre-critical', 'critical', 'post-critical']
NAMES = ['x', 'z', 'epsilon']
NUM_COLS = 3

def main():
	"""Plot the trajectories of particles of critical size and density."""
	numerics = pd.read_csv(IN_FILE)
	amplitudes = numerics['A\''].drop_duplicates().tolist() 
	plot_num = len(amplitudes) * 100 + NUM_COLS * 10 + 1

	# plot pre-critical, critical, post-critical trajectories
	for (i, j), m in zip(product(range(len(amplitudes)), range(NUM_COLS)),
						 range(len(amplitudes) * NUM_COLS)):
		# parameters used to extract data
		st = extract_data('St', numerics, {'A\'': amplitudes[i]})\
			.drop_duplicates().iloc[j]
		params = {'St': st, 'A\'': amplitudes[i], 'wave_type': 'linear'}

		# initialize figure and plot curves
		fig(num=plot_num + m)
		x, z, _ = extract_data(NAMES, numerics, params)
		plt.plot(x, z, '-k', label='linear wave')
		params['wave_type'] = 'stokes'
		x, z, epsilon = extract_data(NAMES, numerics, params)
		plt.plot(x, z, '--k', label='fifth order wave')

		# create label to show the value of epsilon as a fraction of pi
		epsilon = epsilon.iloc[0]
		if Fraction(epsilon / np.pi).limit_denominator(1000).numerator == 1:
			label = rf'$\epsilon = \pi / ${Fraction(epsilon / np.pi)\
					  .limit_denominator(1000).denominator:g}' + r'$\approx$' \
				   + f'{epsilon:.2f}'
		else:
			label = rf'$\epsilon = ${Fraction(epsilon / np.pi)\
					  .limit_denominator(1000).numerator:g}$\pi$'\
				   + f'/{Fraction(epsilon / np.pi).limit_denominator(1000)\
					  .denominator:g}' + r'$\approx$' + f'{epsilon:.1f}'

		# place axis labels
		if i == len(amplitudes) - 1: plt.xlabel(r'$x$')
		if i == len(amplitudes) - 1 and j == NUM_COLS - 1: plt.legend()
		if i == 0: plt.title(TITLES[j])
		if j == 0: plt.ylabel(r'$z$')
		if j == NUM_COLS - 1:
			plt.ylabel(label)
			plt.gca().yaxis.set_label_position('right')
	plt.show()
if __name__ == '__main__': main()

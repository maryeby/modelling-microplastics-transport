import pandas as pd
import matplotlib.pyplot as plt
from numpy import pi

from utils.data_tools import extract_data
from utils.plot import initialize_figure as fig
from examples.linear_wave.particle_velocity.numerics import STOKES_HATS, RS, \
															INCLUDE_HISTORY
from examples.linear_wave.particle_velocity.numerics import OUT_FILE as IN_FILE

def main():
	"""Plot the horizontal velocity of a particle in a wave over time."""
	# read data and initialize figure
	numerics = pd.read_csv(IN_FILE)
	fig(r'$t$', r'$\dot{x}$')

	# plot curves and data for each combination of parameters
	for stokes_hat, r, history in zip(STOKES_HATS, RS, INCLUDE_HISTORY):
		# extract data
		params = {'curve_type': 'xdot', 'Sthat': stokes_hat, 'R': r,
				  'history': history}
		t, xdot = extract_data(['t', 'curve'], numerics, params)
		
		# plot curves
		if history and stokes_hat == STOKES_HATS[0]:
			label = 'with history'
		elif stokes_hat == STOKES_HATS[0]:
			label = 'without history'
		else:
			label = ''
		color = 'k' if stokes_hat == STOKES_HATS[0] else 'silver'
		style = ':' if history else '-'
		plt.plot(t, xdot, c=color, ls=style, label=label)
	t, peaks = extract_data(['t_peaks', 'peaks'], numerics)
	plt.scatter(t, peaks, ec='k', fc='none')
	plt.legend()
	plt.show()

if __name__ == '__main__': main()

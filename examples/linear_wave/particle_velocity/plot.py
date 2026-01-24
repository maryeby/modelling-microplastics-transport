import pandas as pd
import matplotlib.pyplot as plt
from numpy import pi

from utils.data_tools import extract_data
from utils.plot import initialize_figure as fig
from examples.linear_wave.particle_velocity.numerics import STOKES_HATS, RS, \
															INCLUDE_HISTORY
from examples.linear_wave.particle_velocity.numerics import OUT_FILE as IN_FILE

def main():
	"""Plot the horizontal velocity decay of a particle in a wave over time."""
	# read data and initialize figure
	numerics = pd.read_csv(IN_FILE)
	fig(r'$t$', r'$\dot{x}$', lims=[0, 30, 0, 0.08])
	names = ['t', 'decay_curve', 'S', 'R']

	# plot curves and data for each combination of parameters
	for stokes_hat, r, history in zip(STOKES_HATS, RS, INCLUDE_HISTORY):
		# extract data
		params = {'Sthat': stokes_hat, 'R': r, 'history': history}
		t, curve, s, r = extract_data(names, numerics, params)
		s, r = s.iloc[0], r.iloc[0]
		
		# plot curves
		label = '' if history else rf'$S$ = {s:g}, $R$ = {r:g}'
		color = 'k' if stokes_hat == STOKES_HATS[0] else 'silver'
		style = '--' if history else '-'
		plt.plot(t, curve, c=color, ls=style, label=label)
	plt.legend()
	plt.show()

if __name__ == '__main__': main()

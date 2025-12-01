import pandas as pd
import matplotlib.pyplot as plt
from numpy import pi

from utils.data_tools import extract_data
from utils.plot import initialize_figure as fig
from utils.plot import JFM_WIDTH as JFM
from examples.linear_wave.particle_velocity.numerics import STOKES_HATS, RS, \
	 INCLUDE_HISTORY
from examples.linear_wave.particle_velocity.numerics import OUT_FILE as IN_FILE

def main():
	"""Plot the horizontal velocity decay of a particle in a wave over time."""
	# read data and initialize figure
	numerics = pd.read_csv(IN_FILE)
	fig(r'$t$', r'$\dot{x}$', width=JFM / 2)

	# plot curves and data for each combination of parameters
	for stokes_hat, r, history in zip(STOKES_HATS, RS, INCLUDE_HISTORY):
		# extract data
		params = {'curve_type': 'xdot', 'Sthat': stokes_hat, 'R': r,
				  'history': history}
		t = extract_data('t', numerics, params)
		xdot = extract_data('curve', numerics, params)
		params['curve_type'] = 'fitted'
		curve = extract_data('curve', numerics, params)
		
		# plot curves
		color = 'k' if stokes_hat == STOKES_HATS[0] else 'silver'
		style = '--' if history else '-'
		label = 'with history effects' if history else 'without history effects'
		if stokes_hat != STOKES_HATS[0]: label=''
		plt.plot(t, curve, c=color, ls=style, label=label)
	plt.legend()
	plt.show()

if __name__ == '__main__': main()

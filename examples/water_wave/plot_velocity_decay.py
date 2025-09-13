import pandas as pd
import matplotlib.pyplot as plt

from utils.data_tools import extract_data
from utils.plot import initialize_figure as fig
from utils.plot import JFM_WIDTH as JFM
from examples.water_wave.particle_velocity import STOKES_NUMS, BETAS, \
												  INCLUDE_HISTORY, SCALE
from examples.water_wave.particle_velocity import OUT_FILE as IN_FILE

def main():
	"""Plot the horizontal velocity decay over time of a particle in a wave."""
	# read data and initialize figure
	numerics = pd.read_csv(IN_FILE)
	xlim = max(numerics['t'].to_numpy())
	fig(r'$t$', r'$\dot{x}$', lims=[-0.2, xlim + 0.1, -0.1, 1.5], width=JFM / 2)

	# plot curves and data for each combination of parameters
	for stokes_num, beta, history in zip(STOKES_NUMS, BETAS, INCLUDE_HISTORY):
		# extract data
		params = {'curve_type': 'xdot', 'St': stokes_num, 'beta': beta,
				  'history': history}
		t = extract_data('t', numerics, params)
		xdot = extract_data('curve', numerics, params)
		params['curve_type'] = 'fitted'
		curve = extract_data('curve', numerics, params)
		
		# plot curves
		color = 'k' if stokes_num == STOKES_NUMS[0] else 'silver'
		style = '--' if history else '-'
		label = 'with history effects' if history else 'without history effects'
		if stokes_num != STOKES_NUMS[0]: label=''
		plt.plot(t, curve, c=color, ls=style, label=label)

	plt.legend()
	plt.show()

if __name__ == '__main__': main()

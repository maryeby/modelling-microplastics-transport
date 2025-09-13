import pandas as pd
import matplotlib.pyplot as plt

from utils.data_tools import extract_data
from utils.plot import initialize_figure as fig
from examples.water_wave.particle_velocity import STOKES_NUMS, BETAS, \
												  INCLUDE_HISTORY, SCALE
from examples.water_wave.particle_velocity import OUT_FILE as IN_FILE

def main():
	"""Plot the time vs horizontal velocity of a particle in a wave."""
	# read data and initialize figure
	numerics = pd.read_csv(IN_FILE)
	fig(r'$t$', r'$\dot{x}$', width='jfm')

	# plot curves and data for each combination of parameters
	for stokes_num, beta, history in zip(STOKES_NUMS, BETAS, INCLUDE_HISTORY):
		# extract data
		params = {'curve_type': 'xdot', 'St': stokes_num, 'beta': beta,
				  'history': history}
		t = extract_data('t', numerics, params)
		xdot = extract_data('curve', numerics, params)
		
		# plot curves
		if history and stokes_num == STOKES_NUMS[0]:
			label = 'with history'
		elif stokes_num == STOKES_NUMS[0]:
			label = 'without history'
		else:
			label = ''
		color = 'k' if stokes_num == STOKES_NUMS[0] else 'silver'
		style = ':' if history else '-'
		plt.plot(t, xdot, c=color, ls=style, label=label)
		if history: print(f'St = {stokes_num:g}\nR = {beta * SCALE:.2f}\n')
	plt.legend()
	plt.show()

if __name__ == '__main__': main()

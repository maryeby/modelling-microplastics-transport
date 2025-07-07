import pandas as pd
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from examples.water_wave.sensitivity_test import OUT_FILE as IN_FILE

def main():
	"""Plot the timestep sensitivity of the model."""
	numerics = pd.read_csv(IN_FILE)
	delta_x = numerics['delta_x'].to_numpy() / numerics['delta_x'].to_numpy()[1]
	delta_t = numerics['delta_t'].to_numpy() / numerics['delta_t'].to_numpy()[1]

	# plot timestep size vs horizontal displacement
	fig(r'$\Delta t$', r'$\Delta x$', make_square=True)
	plt.plot(delta_t, delta_x, '-ko', data=numerics)
	plt.show()
if __name__ == '__main__': main()

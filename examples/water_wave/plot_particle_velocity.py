import pandas as pd
import matplotlib.pyplot as plt

from utils.data_tools import extract_data
from utils.plot import initialize_figure as fig
from examples.water_wave.particle_velocity import OUT_FILE as IN_FILE

def main():
	"""Plot the time vs horizontal velocity of a particle in a wave."""
	numerics = pd.read_csv(IN_FILE)
	t = extract_data('t', numerics, {'curve_type': 'xdot'}).to_numpy()

	# plot particle velocity over time
	fig(r'$t$', r'$\dot{x}$', lims=[-0.2, t[-1] + 0.1, -1.5, 1.5])
	plot_curve(numerics, 'xdot', '-')
	plot_curve(numerics, 'fitted', '--')
#	plot_curve(numerics, 'envelope', '-.')
	plot_curve(numerics, 'decay', ':')
	plt.scatter('t_peaks', 'peaks', edgecolors='k', facecolors='none',
				data=numerics)
	plt.show()

def plot_curve(df, curve_type, ls):
	"""Plot the specified curve over time."""
	t, curve = extract_data(['t', 'curve'], df, {'curve_type': curve_type})
	plt.plot(t, curve, ls + 'k')

if __name__ == '__main__': main()

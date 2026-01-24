import pandas as pd
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data
from examples.bichromatic_wave.basic_example.numerics import OUT_FILE as IN_FILE

def main():
	"""Plot the position and velocity of a particle in a bichromatic wave."""
	numerics = pd.read_csv(IN_FILE)

	# plot particle trajectory
	fig(r'$x$', r'$z$', make_square=True, equal_aspect=False)
	x, z = extract_data(['x', 'z'], numerics, {'history': False})
	plt.plot(x, z, '-k')
	x, z = extract_data(['x', 'z'], numerics, {'history': True})
	plt.plot(x, z, '--k')
	
	# plot particle velocity over time
	fig(y_label=r'$\dot{x}$', num=211)
	t, xdot = extract_data(['t', 'xdot'], numerics, {'history': False})
	plt.plot(t, xdot, '-k')
	t, xdot = extract_data(['t', 'xdot'], numerics, {'history': True})
	plt.plot(t, xdot, '--k')
	fig(r'$t$', r'$\dot{z}$', 212)
	t, zdot = extract_data(['t', 'zdot'], numerics, {'history': False})
	plt.plot(t, zdot, '-k')
	t, zdot = extract_data(['t', 'zdot'], numerics, {'history': True})
	plt.plot(t, zdot, '--k')
	plt.show()

if __name__ == '__main__': main()

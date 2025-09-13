import pandas as pd
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data
from examples.stokes_wave.numerics import OUT_FILE as IN_FILE

def main():
	"""Plot the trajectory and velocity of a particle moving through a wave."""
	numerics = pd.read_csv(IN_FILE)
	x, z = extract_data(['x', 'z'], numerics, {'wave': 'linear'})

	# plot particle trajectory
	fig(r'$x$', r'$z$', make_square=True, equal_aspect=True)
	plt.plot(x, z, '-k', label='linear wave')
	x, z = extract_data(['x', 'z'], numerics, {'wave': 'stokes'})
	plt.plot(x, z, '--k', label='Stokes 5th order wave')
	plt.legend()
	
	# plot horizontal particle velocity over time
	fig(y_label=r'$\dot{x}$', num=211, hide_xticks=True)
	t, xdot = extract_data(['t', 'xdot'], numerics, {'wave': 'linear'})
	plt.plot(t, xdot, '-k', label='linear wave')
	t, xdot = extract_data(['t', 'xdot'], numerics, {'wave': 'stokes'})
	plt.plot(t, xdot, '--k', label='Stokes 5th order wave')

	# plot vertical particle velocity over time
	fig(r'$t$', r'$\dot{z}$', 212)
	t, zdot = extract_data(['t', 'zdot'], numerics, {'wave': 'linear'})
	plt.plot(t, zdot, '-k', label='linear wave')
	t, zdot = extract_data(['t', 'zdot'], numerics, {'wave': 'stokes'})
	plt.plot(t, zdot, '--k', label='Stokes 5th order wave')
	plt.legend()
	plt.show()

if __name__ == '__main__': main()

import pandas as pd
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from examples.bichromatic_wave.numerics import OUT_FILE as IN_FILE

def main():
	"""Plot the trajectory and velocity of a particle moving through a wave."""
	numerics = pd.read_csv(IN_FILE)

	# plot particle trajectory
	fig(r'$x$', r'$z$', make_square=True, equal_aspect=True)
	plt.plot('x', 'z', '-k', data=numerics)
	
	# plot particle velocity over time
	fig(y_label=r'$\dot{x}$', num=211, hide_xticks=True)
	plt.plot('t', 'xdot', '-k', data=numerics)
	fig(r'$t$', r'$\dot{z}$', 212)
	plt.plot('t', 'zdot', '-k', data=numerics)
	plt.show()

if __name__ == '__main__': main()

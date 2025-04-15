import pandas as pd
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from utils.plot import initialize_subplot as subplot
from examples.water_wave.basic_example.numerics import OUT_FILE as IN_FILE

def main():
	"""Plot the trajectory and velocity of a particle moving through a wave."""
	numerics = pd.read_csv(IN_FILE)

	# plot particle trajectory
	fig('x', 'z', make_square=True, equal_aspect=True)
	plt.plot('x', 'z', '-k', data=numerics)
	
	# plot particle velocity over time
	plt.figure(2)
	subplot(211, y_label=r'$\dot{x}$')
	plt.plot('t', 'xdot', '-k', data=numerics)
	subplot(212, r'$t$', r'$\dot{z}$')
	plt.plot('t', 'zdot', '-k', data=numerics)
	plt.show()

if __name__ == '__main__': main()

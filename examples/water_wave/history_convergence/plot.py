import pandas as pd
import matplotlib.pyplot as plt
from utils.plot import initialize_subplot as subplot

IN_FILE = '../../data/water_wave/history_convergence.csv'

def main():
	"""Plot the convergence of the value of the history force at *t* = 0."""
	numerics = pd.read_csv(IN_FILE) # read data

	# initialize drift velocity figure & left subplot
	plt.figure()
	subplot(211, y_label=r'$H\'(0)_x$', x_scale='log')
	plt.plot('delta_t', 'initial_history_x', '-k.', data=numerics)
	subplot(212, r'$\Delta t$', r'$H\'(0)_z$', x_scale='log')
	plt.plot('delta_t', 'initial_history_z', '-k.', data=numerics)
	plt.show()

if __name__ == '__main__':
	main()

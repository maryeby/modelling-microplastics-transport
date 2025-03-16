import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import initialize_subplot as subplot
from utils.plot import FS
from utils.data_tools import extract_data

IN_FILE1 = '../../data/water_wave/drift_vel_numerics.csv'
IN_FILE2 = '../../data/water_wave/drift_vel_analysis.csv'

def main():
	"""
	Plot the drift velocity of a particle in a wave over time.

	The particle is negatively buoyant, transported through a linear wave of
	deep water. Numerical solutions are shown with and without history effects.
	Curves have been fit to the numerical data.
	"""
	# read data
	numerics = pd.read_csv(IN_FILE1)
	analysis = pd.read_csv(IN_FILE2)

	plt.figure()
	subplot(121, r'$t$', r'$\bar{u}$', make_square=True)
	t, u_bar = extract_data(['t_u', 'u'], analysis, {'history': False})
	plt.plot(t, u_bar, '-k')
	t, u_bar = extract_data(['t_u', 'u'], analysis, {'history': True})
	plt.plot(t, u_bar, ':k')
	t, u_bar = extract_data(['t', 'u_bar'], numerics, {'history': False})
	plt.scatter(t, u_bar, edgecolors='k', facecolors='none')
	t, u_bar = extract_data(['t', 'u_bar'], numerics, {'history': True})
	plt.scatter(t, u_bar, marker='s', edgecolors='k', facecolors='none')

	subplot(122, r'$t$', r'$\bar{w}$', make_square=True)
	t, w_bar = extract_data(['t_w', 'w'], analysis, {'history': False})
	plt.plot(t, w_bar, '-k', label='without history effects')
	t, w_bar = extract_data(['t_w', 'w'], analysis, {'history': True})
	plt.plot(t, w_bar, ':k', label='with history effects')
	t, w_bar = extract_data(['t', 'w_bar'], numerics, {'history': False})
	plt.scatter(t, w_bar, edgecolors='k', facecolors='none',
				label='without history effects')
	t, w_bar = extract_data(['t', 'w_bar'], numerics, {'history': True})
	plt.scatter(t, w_bar, marker='s', edgecolors='k', facecolors='none',
				label='with history effects')
	plt.legend(fontsize=FS)
	plt.show()

if __name__ == '__main__':
	main()

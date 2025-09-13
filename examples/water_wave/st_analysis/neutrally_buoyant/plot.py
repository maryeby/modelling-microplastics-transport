import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data
from examples.water_wave.st_analysis.neutrally_buoyant.numerics import DEPTHS, \
	 WAVELENGTH

IN_FILE1 = '../../../data/water_wave/st_neutral_numerics.csv'
IN_FILE2 = '../../../data/water_wave/st_analytics.csv'
STYLES = ['-', '--', ':']

def main():
	"""
	Plot the drift velocity vs vertical position of particles in a wave.

	Analytical and numerical solutions are shown for neutrally buoyant particles
	in linear waves of arbitrarily deep water. Numerical solutions with and
	without history effects are included. The solutions are averaged over the
	trajectory of the particle.
	"""
	# read data files and initilize figure
	numerics = pd.read_csv(IN_FILE1)
	analytics = pd.read_csv(IN_FILE2)
	fig(r'$\bar{u}$', r'$\bar{\bar{z}}$', equal_aspect=True, make_square=True,
		width='jfm')

	# plot analytical solutions
	for i in range(len(DEPTHS)):
		label = r"$ h' / \lambda' = $" \
			  + f'{str(Fraction(DEPTHS[i] / WAVELENGTH).limit_denominator())}'
		z, u = extract_data(['z/h', 'u_d'], analytics, {'depth': DEPTHS[i]})
		plt.plot(u, z, c='k', ls=STYLES[i], label=label)
	plt.legend()

	# plot numerical solutions
	plt.scatter('u_bar', 'z_bar/h', data=numerics, edgecolors='k',
				facecolors='none') 
	plt.show()

if __name__ == '__main__':
	main()

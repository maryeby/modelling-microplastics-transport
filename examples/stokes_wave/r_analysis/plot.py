import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from fractions import Fraction

from utils.data_tools import extract_data, print_parameter
from utils.plot import initialize_figure as fig
from examples.linear_wave.r_analysis.numerics import RS, STOKES_HAT
from examples.stokes_wave.r_analysis.numerics import AMPLITUDES
from examples.stokes_wave.r_analysis.analysis import OUT_FILE as IN_FILE1
from examples.stokes_wave.r_analysis.numerics import OUT_FILE as IN_FILE2

# positions and format of bubble labels
TEXT_POSITION_X = [0, -0.035, -0.415, 0.423, 0.307]
TEXT_POSITION_Y = [0, -0.25, -0.39, -0.25, -0.78]
MARKERS = ['*', '.', 's', 'd']
FS = 8
PROPERTIES = dict(boxstyle='circle', fc='w', ec='k')

def main():
	"""Plot the drift velocities of particles of varying densities in a wave."""
	# read data files and create variables for data extraction & scaling
	analysis = pd.read_csv(IN_FILE1)
	numerics = pd.read_csv(IN_FILE2)
	print_parameter('Sthat', STOKES_HAT)
	names1 = ['z', 'u']
	names2 = ['z_bar', 'u_d_bar', 'mean_speed', 'epsilon']
	params1 = {'R': RS[1], 'analytical': False, 'history': False}
	params2 = {'R': RS[1], 'history': False, 'A\'': AMPLITUDES[1]}

	# plot negatively & positively buoyant curves, data points, and labels
	fig(r'$\bar{u}_d/\epsilon^2$', r'$\bar{z}$', 121,
		lims=[-0.6, 1, -2.5, 0], make_square=True, add_subplot_labels=True)
	for r, history in product(RS[1:], [False, True]):
		fmt = '--k' if history else '-k'
		params1.update(R=r, history=history)
		params2.update(R=r, history=history)
		z1, u1 = extract_data(names1, analysis, params1)
		plt.plot(u1, z1, fmt)
		z2, u2, mean_speed, _ = extract_data(names2, numerics, params2)
		u2, mean_speed = u2.to_numpy(), mean_speed.to_numpy()
		plt.scatter(u2 - (1 - mean_speed), z2, marker='.', ec='k', fc='none')
		j = RS.index(r)
		plt.text(TEXT_POSITION_X[j], TEXT_POSITION_Y[j], f'{RS[j]:.2g}',
				 bbox=PROPERTIES, fontsize=FS)

	# plot neutrally buoyant analytical solutions
	fig(r'$\bar{u}_d/\epsilon^2 + \bar{U}$', r'$\bar{\bar{z}}$', 122,
		make_square=True, add_subplot_labels=True, lims=[-0.6, 1, -2.5, 0])
	z, u = extract_data(names1, analysis, {'analytical': True})
	plt.plot(u, z, '-k')

	# plot neutrally buoyant numerical solutions
	params2.update(R=RS[0], history=False)
	for a, m in zip(AMPLITUDES, MARKERS):
		# extract data
		params2['A\''] = a
		z, u, mean_speed, epsilon = extract_data(names2, numerics, params2)
		u, mean_speed = u.to_numpy(), mean_speed.to_numpy()
		epsilon = epsilon.iloc[0]

		# create label to show the value of epsilon as a fraction of pi
		if Fraction(epsilon / np.pi).limit_denominator(1000).numerator == 1:
			l = rf'$\epsilon = \pi / ${Fraction(epsilon / np.pi)\
				  .limit_denominator(1000).denominator:g}'
		else:
			l = rf'$\epsilon = ${Fraction(epsilon / np.pi)\
				  .limit_denominator(1000).numerator:g}$\pi$'\
			  + f'/{Fraction(epsilon / np.pi).limit_denominator(1000)\
				  .denominator:g}'
		plt.scatter(u + (3 / 2 - mean_speed), z, marker=m, ec='k', fc='none',
					label=l)
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()

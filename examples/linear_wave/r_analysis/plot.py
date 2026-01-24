import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from fractions import Fraction

from utils.data_tools import extract_data
from utils.plot import initialize_figure as fig
from examples.linear_wave.r_analysis.numerics import RS
from examples.linear_wave.r_analysis.analysis import OUT_FILE as IN_FILE1
from examples.linear_wave.r_analysis.numerics import OUT_FILE as IN_FILE2

# positions and format of bubble labels
TEXT_POSITION_X = [0.345, 0.176, -0.025, 0.44, 0.487]
TEXT_POSITION_Y = [-0.52, -0.61, -0.43, -0.79, -1.39]
FS = 8
PROPERTIES = dict(boxstyle='circle', fc='w', ec='k')

def main():
	"""Plot the drift velocities of particles of varying densities in a wave."""
	# read data files and create variables for data extraction & scaling
	analysis = pd.read_csv(IN_FILE1)
	numerics = pd.read_csv(IN_FILE2)
	names1, names2 = ['z', 'u'], ['z_crossings', 'u_bar']
	params1 = {'R': RS[0], 'analytical': True}
	params2 = {'R': RS[0], 'history': True}

	# plot neutrally buoyant curve, data points, and label
	fig(r'$\bar{u}/\epsilon^2$', r'$\bar{z}$', lims=[-0.15, 1, -2, 0],
		make_square=True)
	analytical_z, analytical_u = extract_data(names1, analysis, params1)
	neutral_z, neutral_u = extract_data(names2, numerics, params2)
	plt.plot(analytical_u, analytical_z, ':k')
	plt.scatter(neutral_u, neutral_z, marker='.', ec='k', fc='none')
	plt.text(TEXT_POSITION_X[0], TEXT_POSITION_Y[0],
			 str(Fraction(RS[0]).limit_denominator(1000)), bbox=PROPERTIES,
			 fontsize=FS)

	# plot negatively & positively buoyant curves, data points, and labels
	params1['analytical'], params1['history'] = False, False
	for r, history in product(RS[1:], [False, True]):
		fmt = '--k' if history else '-k'
		params1.update(R=r, history=history)
		params2.update(R=r, history=history)
		z1, u1 = extract_data(names1, analysis, params1)
		plt.plot(u1, z1, fmt)
		z2, u2 = extract_data(names2, numerics, params2)
		plt.scatter(u2, z2, marker='.', ec='k',
					fc='none')
		j = RS.index(r)
		plt.text(TEXT_POSITION_X[j], TEXT_POSITION_Y[j], f'{RS[j]:.2g}',
				 bbox=PROPERTIES, fontsize=FS)
	plt.show()

if __name__ == '__main__':
	main()

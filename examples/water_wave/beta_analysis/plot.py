import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from fractions import Fraction

from utils.data_tools import extract_data
from utils.plot import initialize_figure as fig
from examples.water_wave.beta_analysis.numerics import AMPLITUDE, WAVELENGTH, \
													   BETAS, SCALE
from examples.water_wave.beta_analysis.analysis import OUT_FILE as IN_FILE1
from examples.water_wave.beta_analysis.numerics import OUT_FILE as IN_FILE2

R = SCALE * np.array(BETAS)

def main():
	"""Plot the drift velocities of particles of varying densities in a wave."""
	# read data files and create variables for data extraction & scaling
	analysis = pd.read_csv(IN_FILE1)
	numerics = pd.read_csv(IN_FILE2)
	names1, names2 = ['z', 'u'], ['z_crossings', 'u_bar']
	params1 = {'beta': BETAS[0], 'analytical': True}
	params2 = {'beta': BETAS[0], 'history': True}
	k = 2 * np.pi / WAVELENGTH

	# positions and format of bubble labels
	text_position_x = [0.345, 0.191, 0.15, 0.487, 0.338]
	text_position_y = [-0.52, -0.22, -0.55, -1.39, -2.68]
	properties = dict(boxstyle='circle', fc='w', ec='k')

	# plot neutrally buoyant curve, data points, and label
	fig(r'$\bar{u}$', r'$\bar{z}$', lims=[-0.075, 1, -7, 0], make_square=True)
	ax = plt.gca()
	analytical_z, analytical_u = extract_data(names1, analysis, params1)
	neutral_z, neutral_u = extract_data(names2, numerics, params2)
	plt.plot(analytical_u, analytical_z, ':k')
	plt.scatter(neutral_u / (k * AMPLITUDE), neutral_z, marker='.',
				ec='k', fc='none')

	# plot negatively buoyant curves, data points, and labels
	params1['analytical'], params1['history'] = False, False
	for beta, history in product(BETAS[3:], [False, True]):
		fmt = '--k' if history else '-k'
		params1.update(beta=beta, history=history)
		params2.update(beta=beta, history=history)
		z1, u1 = extract_data(names1, analysis, params1)
		plt.plot(u1, z1, fmt)
		z2, u2 = extract_data(names2, numerics, params2)
		plt.scatter(u2 / (k * AMPLITUDE), z2, marker='.', ec='k',
					fc='none')
		j = BETAS.index(beta)
		plt.text(text_position_x[j], text_position_y[j],
				 str(Fraction(R[j]).limit_denominator()), bbox=properties,
				 fontsize=8)

	# plot first positively buoyant curve and data points without history
	params1.update(beta=BETAS[1], history=False)
	params2.update(beta=BETAS[1], history=False)
	z3, u3 = extract_data(names1, analysis, params1)
	plt.plot(u3, z3, '-k', label='')
	z4, u4 = extract_data(names2, numerics, params2)
	plt.scatter(u4 / (k * AMPLITUDE), z4, marker='.', ec='k', fc='none',
				label='')

	# plot first positively buoyant curve, data points, and label with history
	params1['history'], params2['history'] = True, True
	z5, u5 = extract_data(names1, analysis, params1)
	plt.plot(u5, z5, '--k', label='')
	z6, u6 = extract_data(names2, numerics, params2)
	plt.scatter(u6 / (k * AMPLITUDE), z6, marker='.', ec='k', fc='none',
				label='')

	# plot second positively buoyant curve and data points without history
	params1.update(beta=BETAS[2], history=False)
	params2.update(beta=BETAS[2], history=False)
	z7, u7 = extract_data(names1, analysis, params1)
	plt.plot(u7, z7, '-k', label='without history effects')
	z8, u8 = extract_data(names2, numerics, params2)
	plt.scatter(u8 / (k * AMPLITUDE), z8, marker='.', ec='k', fc='none',
				label='')

	# plot second positively buoyant curve, data points, and label with history
	params1['history'], params2['history'] = True, True
	z9, u9 = extract_data(names1, analysis, params1)
	plt.plot(u9, z9, '--k', label='with history effects')
	z10, u10 = extract_data(names2, numerics, params2)
	plt.scatter(u10 / (k * AMPLITUDE), z10, marker='.', ec='k',
				fc='none', label='')

	# add inset and plot curves
	axins = ax.inset_axes([0.4, -6.5, 0.56, 3.19], xlim=(-0.075, 0.6),
						  ylim=(-1, 0), transform=ax.transData)
	axins.plot(analytical_u, analytical_z, ':k')
	axins.plot(u3, z3, '-k', label='')
	axins.plot(u5, z5, '--k', label='')
	axins.plot(u7, z7, '-k', label='without history effects')
	axins.plot(u9, z9, '--k', label='with history effects')

	# plot data points in the inset
	axins.scatter(neutral_u / (k * AMPLITUDE), neutral_z, marker='.', ec='k',
				  fc='none')
	axins.scatter(u4 / (k * AMPLITUDE), z4, marker='.', ec='k', fc='none',
				  label='')
	axins.scatter(u6 / (k * AMPLITUDE), z6, marker='.', ec='k', fc='none',
				  label='')
	axins.scatter(u8 / (k * AMPLITUDE), z8, marker='.', ec='k', fc='none',
				  label='')
	axins.scatter(u10 / (k * AMPLITUDE), z10, marker='.', ec='k', fc='none',
				  label='')

	# plot bubble labels in the inset
	axins.text(text_position_x[1], text_position_y[1],
			   str(Fraction(R[1]).limit_denominator()), bbox=properties,
			   fontsize=8)
	axins.text(text_position_x[0], text_position_y[0],
			   str(Fraction(R[0]).limit_denominator()), bbox=properties,
			   fontsize=8)
	axins.text(text_position_x[2], text_position_y[2],
			   str(Fraction(R[2]).limit_denominator()), bbox=properties,
			   fontsize=8)
	ax.indicate_inset_zoom(axins)
#	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()

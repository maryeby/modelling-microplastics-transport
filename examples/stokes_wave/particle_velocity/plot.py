import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

from utils.data_tools import extract_data
from utils.plot import initialize_figure as fig
from examples.stokes_wave.particle_velocity.numerics import STOKES_HATS, RS, \
															AMPLITUDES
from examples.stokes_wave.particle_velocity.analysis import TOL
from examples.linear_wave.particle_velocity.numerics import AMPLITUDE, \
															WAVELENGTH
from examples.linear_wave.particle_velocity.numerics import OUT_FILE as IN_FILE2
from examples.stokes_wave.particle_velocity.analysis import OUT_FILE as IN_FILE1

LINEAR_EPSILON = 2 * pi * AMPLITUDE / WAVELENGTH

def main():
	"""Plot the horizontal velocity decay of a particle in a wave over time."""
	analysis = pd.read_csv(IN_FILE1)
	linear = pd.read_csv(IN_FILE2)
	fig(r'$\epsilon$', r'$\delta$')
	names = ['S', 'epsilon', 'decay_rate', 'R^2']
	for h in [False, True]:
		# extract data
		params = {'Sthat': STOKES_HATS[0], 'R': RS[0], 'history': h}
		s, epsilon, delta, rsq = extract_data(names, analysis, params)
		linear_delta = extract_data('decay_rate', linear, params).iloc[0]
		s = s.iloc[0]

		# plot curves
		style = '--' if h else '-'
		plt.plot(epsilon, delta, c='k', ls=style, marker='.')
		plt.scatter(LINEAR_EPSILON, linear_delta, c='k', marker='d')
		plt.scatter(epsilon.where(rsq < TOL), delta.where(rsq < TOL), ec='k',
					fc='none')
	plt.show()

if __name__ == '__main__': main()

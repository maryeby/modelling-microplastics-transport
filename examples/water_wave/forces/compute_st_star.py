import numpy as np
import pandas as pd
from intersect import intersection
from itertools import product

from utils.data_tools import extract_data, update_results
from examples.water_wave.forces.numerics import AMPLITUDE, SCALE, STOKES_NUMS

BETAS = [0.9, 0.99, 1.03]
WAVELENGTHS = np.array([np.pi, 4 * np.pi / 5, 2 * np.pi / 3, 4 * np.pi / 7,
						np.pi / 2, 4 * np.pi / 9, 2 * np.pi / 5])
EPSILONS = np.round(2 * np.pi * AMPLITUDE / WAVELENGTHS, 5)

NUM_FILES = 21
IN_FILES = ['../../data/water_wave/st_star/coeffs' + str(i) + '.csv' \
			for i in range(NUM_FILES)]
OUT_FILE = '../../data/water_wave/st_star/analysis.csv'

def main():
	r"""
	Compute $St^*$, where the history force overcomes the Stokes drag.

	The influence of the history force on a particle in a linear wave overcomes
	the influence of the Stokes drag when the amplitude of the horizontal
	component of the history force intersects the amplitude of the horizontal
	component of the Stokes drag, as shown in subplot *(a)* of the figure
	produced by `plot_coefficients.py`. This intersection value is referred to
	as $St^*$, and is computed for various values of the wave steepness
	$\epsilon$ and various particle densities.
	"""
	results = {'St*': [], 'epsilon': [], 'beta': [], 'R': []}
	params = {'method': 'curve_fit'}
	pairs = [x for x in product(BETAS, EPSILONS)]
	for i in range(NUM_FILES):
		numerics = pd.read_csv(IN_FILES[i])
		params['force'] = 'stokes_drag'
		stokes_drag = extract_data('A', numerics, params)
		params['force'] = 'history_force'
		history = extract_data('A', numerics, params)
		st_star, a_star, _, _ = intersection(STOKES_NUMS, stokes_drag,
											 STOKES_NUMS, history)
		st_star, a_star = st_star[0], a_star[0]
		beta, epsilon = pairs[i]
		r = beta * SCALE
		results = update_results(results, [], [st_star, epsilon, beta, r])
	pd.DataFrame(results).to_csv(OUT_FILE)

if __name__ == '__main__':
	main()

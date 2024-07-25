import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
			  + 'modelling-microplastics-transport')
import numpy as np
import pandas as pd
from examples.water_wave.numerics import single_simulation as run

def main():
	"""
	This program computes numerical solutions for the history force at various
	time step sizes and saves the results to the `data/water_wave` directory.
	"""
	filepath = '../../data/water_wave/history_convergence.csv'

	# initialize variables for the simulations and to store results
	position, St, beta = (0, 0), 0.01, 0.9
	h, A, wavelength = 10, 0.02, 1
	delta_ts = np.linspace(5e-3, 3e-4, 10)
	initial_history_x, initial_history_z = [], []

	# run simulations with various time step sizes
	for delta_t in delta_ts:
		num_periods = delta_t * 8
		print(f'delta_t = {delta_t:.2e}')
		_, history = run(position, St, h, A, wavelength, beta, num_periods, 
						 delta_t, filepath, mode='r',
						 crop=['history_force_x', 'history_force_z'])

		# store results to appropriate lists
		history_x, history_z = history
		initial_history_x.append(history_x[0])
		initial_history_z.append(history_z[0])
		print()

	# store results in a DataFrame and write to the data file
	results = pd.DataFrame({'delta_t\'': delta_ts,
							'H\'(0)_x': initial_history_x,
							'H\'(0)_z': initial_history_z})
	results.to_csv(filepath, index=False)

if __name__ == '__main__':
	main()

import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
from scipy import constants

from models import my_system as ts
DATA_PATH = '../../../data/water_wave/'

def main():
	"""
	This program computes numerical and analytical solutions for the Stokes
	drift velocity of a negatively buoyant inertial particle in a linear water
	wave and saves the results to the `data/water_wave` directory.
	"""
	x_0, z_0 = 0, 0
	# initialize variables
	St, beta = 0.01, 0.9
	h, A, wavelength = 10, 0.02, 1 # wave parameters
	delta_t = 5e-3
	R = 2 / 3 * beta

	# read numerical data
	in_file = 'numerics.csv'
	out_file = 'santamaria_fig2_recreation.csv'
	numerics = pd.read_csv(DATA_PATH + in_file)

	# create condition and lambdas to help filter through numerical data
	condition = (numerics['x_0'] == x_0) & (numerics['z_0'] == z_0) \
									& (numerics['St'] == St) \
									& (numerics['beta'] == beta) \
									& (numerics['h\''] == h) \
									& (numerics['A\''] == A) \
									& (numerics['wavelength\''] == wavelength) \
									& (numerics['delta_t\''] == delta_t)
	get_single = lambda name : numerics[name].where(condition).dropna().iloc[0]
	get_series = lambda name, history : numerics[name].where(condition \
											& (numerics['history'] == history))\
													  .dropna().to_numpy()
	# retrieve relevant numerical results
	U = get_single('U\'')
	k = get_single('k\'')
	Fr = get_single('Fr')

	x = get_series('x', False)
	z = get_series('z', False)
	xdot = get_series('xdot', False)
	t = get_series('t', False)

	x_history = get_series('x', True)
	z_history = get_series('z', True)
	xdot_history = get_series('xdot', True)
	t_history = get_series('t', True)

	# compute drift velocity
	x_crossings, z_crossings, u_d, w_d, t_d = ts.compute_drift_velocity(x, z,
																		xdot, t)
	x_crossings_history, z_crossings_history, \
		u_d_history, w_d_history, t_d_history = ts.compute_drift_velocity(
												   x_history, z_history,
												   xdot_history, t_history)
	# scale results
	t_d /= Fr
	u_d /= Fr
	w_d *= U ** 2 * k * R / (constants.g * St)
	
	t_d_history /= Fr
	u_d_history /= Fr
	w_d_history *= U ** 2 * k * R / (constants.g * St)

	# write results to data file
	history = [True] * len(x_crossings)
	no_history = [False] * len(x_crossings)
	results = {'x_crossings': x_crossings, 'z_crossings': z_crossings,
			   'u_d': u_d, 'w_d': w_d, 't': t_d, 'history': no_history}
	results_history = {'x_crossings': x_crossings_history,
					   'z_crossings': z_crossings_history,
					   'u_d': u_d_history, 'w_d': w_d_history, 't': t_d_history,
					   'history': history}
	results = dict([(key, pd.Series(value)) for key, value in results.items()])
	results_history = dict([(key, pd.Series(value)) for key, value
							in results_history.items()])
	df1, df2 = pd.DataFrame(results), pd.DataFrame(results_history)
	df = pd.concat([df1, df2])
	df.to_csv(DATA_PATH + out_file, index=False)

if __name__ == '__main__':
	main()

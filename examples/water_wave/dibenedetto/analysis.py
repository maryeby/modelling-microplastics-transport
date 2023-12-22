import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
from scipy import constants

from models import my_system as ts
DATA_PATH = '../../data/water_wave/'

def main():
	"""
	This program computes numerical and analytical solutions for the Stokes
	drift velocity of a negatively buoyant inertial particle in a linear water
	wave and saves the results to the `data/water_wave` directory.
	"""
	# initialize variables
	St, beta_SM = 0.01, 0.9
	h, A, wavelength = 50, 0.02, 1 # wave parameters
	delta_t = 5e-3
	R = 2 / 3 * beta_SM

	# read numerical data
	in_file = 'numerics.csv'
	out_file = 'santamaria_fig2_recreation.csv'
	numerics = pd.read_csv(DATA_PATH + in_file)

	# create conditions to help filter through numerical data
	history = (numerics['St'] == St) & (numerics['beta'] == beta_SM) \
									 & (numerics['history'] == True) \
									 & (numerics['h'] == h) \
									 & (numerics['A'] == A) \
									 & (numerics['wavelength'] == wavelength) \
									 & (numerics['delta_t'] == delta_t)
	no_history = (numerics['St'] == St) & (numerics['beta'] == beta_SM) \
									& (numerics['history'] == False) \
									& (numerics['h'] == h) \
									& (numerics['A'] == A) \
									& (numerics['wavelength'] == wavelength) \
									& (numerics['delta_t'] == delta_t)

	# retrieve relevant numerical results
	U = numerics['U'].where(no_history).dropna().iloc[0]
	k = numerics['k'].where(no_history).dropna().iloc[0]
	h = numerics['h'].where(no_history).dropna().iloc[0]
	Fr = numerics['Fr'].where(no_history).dropna().iloc[0]

	x = numerics['x'].where(no_history).dropna().to_numpy()
	z = numerics['z'].where(no_history).dropna().to_numpy()
	xdot = numerics['xdot'].where(no_history).dropna().to_numpy()
	t = numerics['t'].where(no_history).dropna().to_numpy()

	x_history = numerics['x'].where(history).dropna().to_numpy()
	z_history = numerics['z'].where(history).dropna().to_numpy()
	xdot_history = numerics['xdot'].where(history).dropna().to_numpy()
	t_history = numerics['t'].where(history).dropna().to_numpy()

	# compute numerical solutions for the drift velocity
	x_crossings, z_crossings, u_d, w_d, t_d = ts.compute_drift_velocity(x, z,
																		xdot, t)
	x_crossings_history, z_crossings_history, \
		u_d_history, w_d_history, t_d_history = ts.compute_drift_velocity(
												   x_history, z_history,
												   xdot_history, t_history)
	# compute analytical solutions for the drift velocity
	z_star = z_crossings[0]
	z_star_history = z_crossings_history[0]
	beta = R * (3 * R + 2) / (2 + R * (3 * R - 1))
	St_lin = omega * St * (beta_SM + ((2 - R) / (2 * R)))

	relative_magnitude = np.sqrt((1 - St_lin ** 2 * (1 - beta)) ** 2 \
									+ (St_lin * (1 - beta)) ** 2)
	settling_velocity = St_lin * (1 - beta) / np.tanh(k * h)
	u_SD = k * A / np.tanh(k * h) * np.cosh(2 * (z_star + k * h)) \
				 / (2 * (np.cosh(k * h)) ** 2)
	du_SD = k * A / np.tanh(k * h) * np.sinh(2 * (z_star + k * h)) \
				  / (np.cosh(k * h)) ** 2

	v_x_drift = relative_magnitude / (1 + settling_velocity ** 2) * u_SD
	v_y_drift = -settling_velocity * (1 + relative_magnitude ** 2
				/ (1 + settling_velocity ** 2) * u_SD
				+ 1 / 2 * np.tanh(k * h) * du_SD) 

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

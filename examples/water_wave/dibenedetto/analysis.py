import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np
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
	x_0, z_0 = 0, 0
	St, beta = 0.01, 0.9
	h, A, wavelength = 10, 0.02, 1 # wave parameters
	delta_t = 5e-3

	# read numerical data
	in_file = 'numerics.csv'
	out_file = 'dibenedetto_analysis.csv'
	numerics = pd.read_csv(DATA_PATH + in_file)

	# create conditions and lambdas to help filter through numerical data
	condition = (numerics['x_0'] == x_0) & (numerics['z_0'] == z_0) \
                                    & (numerics['St'] == St) \
                                    & (numerics['beta'] == beta) \
                                    & (numerics['h\''] == h) \
                                    & (numerics['A\''] == A) \
                                    & (numerics['wavelength\''] == wavelength) \
                                    & (numerics['delta_t\''] == delta_t)
	get_single = lambda name : numerics[name].where(condition).dropna().iloc[0]
	get_series = lambda name, history : numerics[name].where(condition & \
										(numerics['history'] == history))\
													.dropna().to_numpy()
	# retrieve relevant numerical results
	k = get_single('k\'')
	h = get_single('h\'')
	U = get_single('U\'')
	omega = get_single('omega\'')
	amplitude = get_single('A\'')
	kA = k * amplitude

	x = get_series('x', False)
	z = get_series('z', False)
	xdot = get_series('xdot', False)
	t = get_series('t', False)

	x_history = get_series('x', True)
	z_history = get_series('z', True)
	xdot_history = get_series('xdot', True)
	t_history = get_series('t', True)

	# compute numerical solutions for the drift velocity
	x_p0, z_p0, u_d, w_d, t_d = ts.compute_drift_velocity(x, z, xdot, t)
	x_p0_history, z_p0_history, u_d_history, w_d_history, \
				  t_d_history = ts.compute_drift_velocity(x_history, z_history,
													   xdot_history, t_history)
	# compute analytical solutions for the drift velocity
	St_lin = 3 * St / (2 * beta * kA)
	epsilon = kA / np.tanh(k * h)
	A = np.sqrt((1 - St_lin ** 2 * (1 - beta)) ** 2 \
				   + (St_lin * (1 - beta)) ** 2)	# relative magnitude
	v_s_lin = St_lin * (1 - beta) / np.tanh(k * h)  # settling velocity

	u_SD = epsilon ** 2 * np.cosh(2 * (z_p0 + k * h)) \
				   / (2 * (np.cosh(k * h)) ** 2)
	du_SD = epsilon ** 2 * np.sinh(2 * (z_p0 + k * h)) \
					/ (np.cosh(k * h) ** 2)
	u_SD_history = epsilon ** 2 * np.cosh(2 * (z_p0_history + k * h)) \
						   / (2 * (np.cosh(k * h)) ** 2)
	du_SD_history = epsilon ** 2 * np.sinh(2 * (z_p0_history + k * h)) \
							/ (np.cosh(k * h) ** 2)

	v_x_drift = A ** 2 / (1 + v_s_lin ** 2) * u_SD
	v_y_drift = -v_s_lin * (1 + A ** 2 / (1 + v_s_lin ** 2) * u_SD
						 + 1 / 2 * np.tanh(k * h) * du_SD) 
	v_x_drift_history = A ** 2 / (1 + v_s_lin ** 2) * u_SD_history
	v_y_drift_history = -v_s_lin * (1 + A ** 2 / (1 + v_s_lin ** 2)
								 * u_SD_history + 1 / 2 * np.tanh(k * h)
								 * du_SD_history)
	# scale results
	u_d *= kA
	w_d *= kA
	u_d_history *= kA
	w_d_history *= kA

	# write results to data file
	history = [True] * len(x_p0)
	no_history = [False] * len(x_p0)
	results = {'x_p0': x_p0[1:], 'z_p0': z_p0[1:], 'u_d': u_d, 'w_d': w_d,
			   't': t_d, 'v_x_drift': v_x_drift[1:], 'v_y_drift': v_y_drift[1:],
			   'history': no_history}
	results_history = {'x_p0': x_p0_history[1:], 'z_p0': z_p0_history[1:],
					   'u_d': u_d_history, 'w_d': w_d_history, 't': t_d_history,
					   'v_x_drift': v_x_drift_history[1:],
					   'v_y_drift': v_y_drift_history[1:], 'history': history}
	results = dict([(key, pd.Series(value)) for key, value in results.items()])
	results_history = dict([(key, pd.Series(value)) for key, value
							in results_history.items()])
	df1, df2 = pd.DataFrame(results), pd.DataFrame(results_history)
	df = pd.concat([df1, df2])
	df.to_csv(DATA_PATH + out_file, index=False)

if __name__ == '__main__':
	main()

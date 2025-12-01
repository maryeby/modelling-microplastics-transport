import pandas as pd
import numpy as np
from scipy import constants

from utils.data_tools import extract_data, update_results
from models import linear_wave as fl
from models.my_system import compute_drift_velocity
from examples.linear_wave.drift_velocity.numerics import DEPTH, AMPLITUDE, \
	 WAVELENGTH, STOKES_NUM, BETA
from examples.linear_wave.drift_velocity.numerics import OUT_FILE as IN_FILE

KEYS = ['t', 'u_double_bar', 'w_double_bar', 'v_x_drift', 'v_y_drift',
		'v_s_lin', 'history']
OUT_FILE = '../../../data/linear_wave/dibenedetto_analytics.csv'

def main():
	"""
	Compute analytical solutions[^1] for the Stokes drift velocity.

	The Stokes drift velocity is analytically computed for a negatively buoyant
	particle in a linear wave of deep water. The computations follow the
	approach outlined in [1], especially equation (2.12). Results are saved to
	the `data/linear_wave` directory.

	References
	----------
	[^1]: [M. H. DiBenedetto et al. (2022).](
		  https://doi.org/10.1017/jfm.2022.95) Enhanced settling and dispersion
		  of inertial particles in surface waves. *Journal of Fluid Mechanics*
		  936, A38.
	"""
	# read data and get wave conditions for scaling
	numerics = pd.read_csv(IN_FILE)
	wave = fl.LinearWave(DEPTH, AMPLITUDE, WAVELENGTH)
	omega = wave.angular_freq
	k = wave.wavenum

	# create dictionary to store solutions and list of names to extract data
	results = {key: [] for key in KEYS}
	names = ['t', 'x', 'z', 'u_bar', 'w_bar']

	# define variables used to compute analytical drift velocity
	St_lin = 3 * STOKES_NUM / (2 * BETA * k * AMPLITUDE)
	epsilon = k * AMPLITUDE / np.tanh(k * DEPTH)
	A = np.sqrt((1 - St_lin ** 2 * (1 - BETA)) ** 2 \
				   + (St_lin * (1 - BETA)) ** 2)		# relative mag
	v_s_lin = St_lin * (1 - BETA) / np.tanh(k * DEPTH)	# settling vel

	for history in [False, True]:
		# get data from numerics file
		data = extract_data(names, numerics, {'history': history})
		data = [s.to_numpy() for s in data]
		t, x_p0, z_p0, u_bar, w_bar = data

		# compute analytical solutions for the drift velocity
		u_SD = epsilon ** 2 * np.cosh(2 * (z_p0 + k * DEPTH)) \
					   / (2 * (np.cosh(k * DEPTH)) ** 2)
		du_SD = epsilon ** 2 * np.sinh(2 * (z_p0 + k * DEPTH)) \
						/ (np.cosh(k * DEPTH) ** 2)
		v_x_drift = A ** 2 / (1 + v_s_lin ** 2) * u_SD
		v_y_drift = -v_s_lin * (1 + A ** 2 / (1 + v_s_lin ** 2) * u_SD
							 + 1 / 2 * np.tanh(k * DEPTH) * du_SD) 
		# scale results
		u_bar *= k * AMPLITUDE
		w_bar *= k * AMPLITUDE

		# wave average numerical drift velocities again (double bar)
		u_double_bar, w_double_bar = [], []
		for i in range(len(u_bar) - 1):
			u_double_bar.append(np.mean([u_bar[i], u_bar[i + 1]]))
			w_double_bar.append(np.mean([w_bar[i], w_bar[i + 1]]))
		u_double_bar = np.array(u_double_bar)
		w_double_bar = np.array(w_double_bar)
		results = update_results(results, [t[:-1], u_double_bar, w_double_bar,
								 v_x_drift[:-1], v_y_drift[:-1]],
								[v_s_lin, history])
		results = update_results(results, [], [t[-1], None, None, v_x_drift[-1],
								 v_y_drift[-1], v_s_lin, history])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)

if __name__ == '__main__':
	main()

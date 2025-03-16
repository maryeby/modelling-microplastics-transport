import warnings
import pandas as pd
import numpy as np
from scipy import constants
from scipy.optimize import curve_fit

from utils.data_tools import extract_data, update_results
from models.my_system import compute_drift_velocity

IN_FILE = '../../data/water_wave/drift_vel_numerics.csv'
OUT_FILE = '../../data/water_wave/drift_vel_analysis.csv'

def main():
	"""
	Fit curves to numerical solutions for the drift velocity of a particle.

	Curves are fit to numerical solutions for the period-averaged Stokes drift
	velocity of a negatively buoyant particle in a linear wave of deep water.
	Results are saved to the `data/water_wave` directory.
	"""
	warnings.filterwarnings('ignore')
	numerics = pd.read_csv(IN_FILE)
	results = {'t_u': [], 'u': [], 't_w': [], 'w': [], 'history': []}

	for history in [False, True]:
		# get numerical results
		t, u, w = extract_data(['t', 'u_bar', 'w_bar'], numerics,
							   {'history': history})
		t = t.to_numpy()
		u = u.to_numpy()
		w = w.to_numpy()
		# fit curve to horizontal drift velocity
		f = lambda x, a, b, c, d : a * np.exp(b * x) + c * np.exp(d * x)
		coefficients, covariance = curve_fit(f, u, t)
		a, b, c, d = coefficients
		extended_u = np.linspace(u[0], u[-1], 100)
		t_u = f(extended_u, a, b, c, d)
		
		# fit curve to vertical drift velocity
		f = lambda x, a, b, c : a ** np.log(b * x + c)
		coefficients, covariance = curve_fit(f, w, t)
		a, b, c = coefficients
		extended_w = np.linspace(w[0], w[-1], 100)
		t_w = f(extended_w, a, b, c)

		# store results
		results = update_results(results, [t_u, extended_u, t_w, extended_w],
								[history])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

if __name__ == '__main__':
	main()

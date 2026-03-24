import warnings
import pandas as pd
import numpy as np
from itertools import repeat, product
from parallelbar import progress_starmap

from utils.data_tools import extract_data, update_results, print_parameter
from utils.colors import print_failure
from transport_framework import particle as prt
from models import linear_wave as lfl
from models import stokes_wave as sfl
from models import my_system as ts
from models.my_system import compute_drift_velocity as find_crossings
from examples.linear_wave.critical_nums.analysis import DEPTH, WAVELENGTH, \
	 X_0, NUM_PERIODS, DELTA_T
from examples.stokes_wave.critical_nums.analysis import OUT_FILE as IN_FILE

Z_0 = 0
R = 0.6
PRE_C, POST_C = np.round(9 / 70, 5), np.round(60 / 7, 5) # Stokes numbers
INCLUDE_HISTORY = False
NUM_CPUS = None
TIMEOUT = 600
HIDE_PROGRESS = True
KEYS = ['x', 'z', 'x_crossings', 'z_crossings', 'Sthat', 'St', 'R', 'A\'',
		'epsilon', 'wave_type']
OUT_FILE = '../../data/stokes_wave/critical_trajectories.csv'

def main():
	"""
	Compute numerical solutions for particles in a 5th order Stokes wave.

	Simulations are run for negatively buoyant particles in fifth order Stokes
	waves of deep water, for the purposes of illustrating oscillatory damping
	of the particle trajectories. Results are saved to the `data/stokes_wave`
	directory.
	"""
	# create dict to store sols and extract Stokes numbers from data file
	analysis = pd.read_csv(IN_FILE)
	amplitudes = analysis['A\''].drop_duplicates().tolist()
	results = {key: [] for key in KEYS}

	# create lists of Sthats, amplitudes, and wave types for parallel processing
	st_hats, repeated_amplitudes, wave_types = [], [], []
	params = {'R_c': R, 'history': INCLUDE_HISTORY, 'A\'': None}
	for a in amplitudes:
		params['A\''] = a
		st_c = extract_data('St_c', analysis, params).iloc[0]
		st_hats += [PRE_C, st_c, POST_C] * 2
		repeated_amplitudes += [a] * 6
		wave_types += ['linear'] * 3 + ['stokes'] * 3

		# print an error message if the post-critical Stokes number <= St_c
		if POST_C < st_c:
			print_failure('Post-critical Stokes number is too small')
			print('A\' = ', a)
			quit()
	params = zip(wave_types, st_hats, repeated_amplitudes)

	# run simulations in parallel and store results
	warnings.filterwarnings('ignore')
	sols = progress_starmap(run_numerics, params, n_cpu=NUM_CPUS,
							total=len(wave_types), process_timeout=TIMEOUT)
	for sol in sols: results = update_results(results, sol[:4], sol[4:])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

def run_numerics(wave_type, stokes_hat, amplitude):
	r"""
	Run a numerical simulation with the specified $\widehat{St}$.

	Parameters
	----------
	wave_type : str
		The type of wave through which the particle is transported.
	stokes_hat : float
		The Stokes number to use for the initialization of the particle.
	amplitude : float
		The amplitude $A'$ of the wave.

	Returns
	-------
	list
		A list containing the particle positions, $\widehat{St}$, $St$, and $R$.
	"""
	wave = lfl.LinearWave(DEPTH, amplitude, WAVELENGTH) if \
		   wave_type == 'linear' else sfl.StokesWave(DEPTH, amplitude,
													 WAVELENGTH)
	particle = prt.Particle(stokes_hat)
	system = ts.MyTransportSystem(particle, wave, R)

	xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
	y = [X_0, Z_0, xdot_0, zdot_0]
	t = np.arange(0, NUM_PERIODS * wave.period, DELTA_T)
	x, z, xdot, _, t = system.maxey_riley(t, y, INCLUDE_HISTORY,
										  HIDE_PROGRESS)[:5]
	x_crossings, z_crossings, _, _, _ = find_crossings(x, z, xdot, t)
	diff = len(x) - len(x_crossings)
	x_crossings = np.array(x_crossings.tolist() + [np.nan] * diff)
	z_crossings = np.array(z_crossings.tolist() + [np.nan] * diff)
	return [x, z, x_crossings, z_crossings, stokes_hat, system.stokes_num, R,
			amplitude, wave.steepness, wave_type]

if __name__ == '__main__': main()

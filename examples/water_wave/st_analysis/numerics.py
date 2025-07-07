import warnings
import numpy as np
import pandas as pd
from itertools import product, repeat
from parallelbar import progress_starmap

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import water_wave as fl
from models import my_system as ts

# particle conditions
STOKES_NUMS = [0.01, 0.1, 0.3, 0.25, 1]
X_0 = 0

# wave conditions
DEPTH = 10
AMPLITUDE = 0.02
WAVELENGTH = 1.5

# simulation conditions
SCALE = 2 / 3
BETAS = [0.99, 1.01]
DELTA_T = 5e-3
NUM_CPUS = None
TIMEOUT = 600
HIDE_PROGRESS = True
OUT_FILE = '../../data/water_wave/st_numerics.csv'

def main():
	"""
	Run numerical simulations for negatively buoyant particles in a wave.

	Simulations are run with and without history effects, with various Stokes
	numbers, and in linear waves of deep water. The period-averaged Stokes
	drift velocity is also computed. Results are saved to the `data/water_wave`
	directory.

	See Also
	--------
	models.my_system.compute_drift_velocity
	"""
	# create dict to store sols, initialize variables for the simulations
	keys = ['t', 'x', 'z', 'u_bar', 'w_bar', 'St', 'beta', 'history']
	results = {key: [] for key in keys}
	wave = fl.WaterWave(DEPTH, AMPLITUDE, WAVELENGTH)

	# create parameters to run simulations in parallel
	repeated_st = [STOKES_NUMS[0]] + STOKES_NUMS[-2:] + STOKES_NUMS[:3]
	repeated_betas = [BETAS[0]] * 3 + [BETAS[1]] * 3
	repeated_history = [False] * len(repeated_st) + [True] * len(repeated_st)
	repeated_st += repeated_st
	repeated_betas += repeated_betas
	params = zip(repeat(wave), repeated_st, repeated_betas, repeated_history)

	# run simulations in parallel
	warnings.filterwarnings('ignore')
	sols = progress_starmap(run_numerics, params, process_timeout=TIMEOUT,
							n_cpu=NUM_CPUS, total=len(repeated_st))
	# store results
	for sol in sols:
		update_results(results, sol[:3], [None, None] + sol[-3:])
		update_results(results, sol[3:-3], sol[-3:])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

def run_numerics(wave, stokes_num, beta, include_history):
	"""
    Run a numerical simulation and compute the horizontal drift velocity.

    Parameters
    ----------
    wave : Wave (obj)
        The wave through which the particle is transported.
	stokes_num : float
		The Stokes number *St*.
    beta : float
        The ratio between the particle and fluid densities.
    include_history : bool
        Whether to include history effects.

    Returns
    -------
    list
        A list containing the vertical particle positions at each period, the
        horizontal drift velocity, the density ratio, and whether history
        effects were included.
    """
	# create Particle and TransportSystem objects, time series data
	particle = prt.Particle(stokes_num)
	system = ts.MyTransportSystem(particle, wave, SCALE * beta)
	num_periods = 70 if stokes_num == STOKES_NUMS[0] else 20
	t = np.arange(0, wave.period * num_periods, DELTA_T)

	# set initial position and velocity of the particle
	z_0 = 0 if beta < 1 else -2
	xdot_0, zdot_0 = wave.velocity(X_0, z_0, t=0)
	y = [X_0, z_0, xdot_0, zdot_0]

	# run simulation and compute drift velocity
	x, z, xdot, _, t = system.maxey_riley(t, y, include_history,
										  HIDE_PROGRESS)[:5]
	x_cross, z_cross, u, w, t_cross = ts.compute_drift_velocity(x, z, xdot, t)
	return[t, x, z, t_cross[1:], x_cross[1:], z_cross[1:], u, w, stokes_num,
		   beta, include_history]

if __name__ == '__main__':
	main()

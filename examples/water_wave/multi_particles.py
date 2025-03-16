import warnings
import numpy as np
import pandas as pd
from itertools import repeat
from parallelbar import progress_starmap
from tqdm import trange
#import matplotlib.pyplot as plt
#from utils.plot import initialize_figure as fig

from utils.data_tools import update_results
from transport_framework import particle as prt 
from models import water_wave as fl
from models import my_system as ts

# particle conditions
STOKES_NUM = 0.01
NUM_POINTS = 10

# wave conditions
DEPTH = 10
AMPLITUDE = 0.02
WAVELENGTH = 1
SHIFT_COEFF = 1 / 8

# simulation conditions
SCALE = 2 / 3
BETA = 0.9
R = SCALE * BETA
NUM_PERIODS = 13
DELTA_T = 5e-3
HIDE_PROGRESS = True
OUT_FILE = '../data/water_wave/multi_particles.csv'

def main():
	"""
	Simulate particles in a wave and periodically record their positions.

	The particles simulated are negatively buoyant, and are transported through
	a linear wave of arbitrarily deep water. The initial positions of the
	particles vary, but their density and size are equivalent. Conditions for
	the simulations were chosen in accordance with [1] Figure 5. Results are
	saved to the `data/water_wave` directory.

	References
	----------
	[^1]: [M. H. DiBenedetto et al. (2022).](
		  https://doi.org/10.1017/jfm.2022.95) Enhanced settling and dispersion
		  of inertial particles in surface waves. *Journal of Fluid Mechanics*
		  936, A38.
	"""
	# create the Particle, Flow, and TransportSystem objects
	particle = prt.Particle(STOKES_NUM)
	wave = fl.WaterWave(DEPTH, AMPLITUDE, WAVELENGTH)
	system = ts.MyTransportSystem(particle, wave, R)

	# create dict to store sols, initialize variables for the simulations
	results = {'x': [], 'z': [], 'x_0': [], 'z_0': [], 'history': []}
	t = np.arange(0, NUM_PERIODS * wave.period, DELTA_T)
	x_0s, z_0s = [], []
	warnings.filterwarnings('ignore')

	# find indices of data points to save
	period = int((1 + SHIFT_COEFF) * wave.period / DELTA_T)
	shift = int(SHIFT_COEFF * wave.period / DELTA_T)
	indices = []
	for i in range(NUM_PERIODS - 1): indices.append(i * period)

	# run first simulation for x_0 = (0, 0)
	sols = run_numerics(system, t, x_0=0, z_0=0, include_history=False)
	x, z = sols[:2]
#	fig('x', 'z', equal_aspect=True, make_square=True)
#	plt.plot(x, z, c='k')

	# compute other initial positions
	n = period // NUM_POINTS
	for i in range(1, NUM_POINTS):
		x_0s.append(x[i * n + shift])
		z_0s.append(z[i * n + shift])

	# store solutions
	x = x[indices]
	z = z[indices]
	results = update_results(results, [x, z], sols[2:])

	# run non-history simulations in parallel and store solutions
	params = zip(repeat(system), repeat(t), x_0s, z_0s, repeat(False))
	sols = progress_starmap(run_numerics, params, n_cpu=4, total=NUM_POINTS)
	for sol in sols:
		sol[0] = sol[0][indices]
		sol[1] = sol[1][indices]
		results = update_results(results, sol[:2], sol[2:])

	# run simulations with history and store results
	x_0s.insert(0, 0)
	z_0s.insert(0, 0)
#	plt.scatter(x_0s, z_0s, marker='.', c='k')
#	plt.show()
#	quit()
	for i in trange(len(x_0s)):
		sols = run_numerics(system, t, x_0s[i], z_0s[i], include_history=True)
		sols[0] = sols[0][indices]
		sols[1] = sols[1][indices]
		results = update_results(results, sols[:2], sols[2:])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

def run_numerics(system, t, x_0, z_0, include_history):
	"""
	Run a numerical simulation with the specified initial position.

	Parameters
	----------
	system : TransportSystem (obj)
		The transport system through which the simulation is run.
	t : ndarray
		1D array containing `float` time series data.
	x_0, z_0 : float
		The initial horizontal and vertical particle position.
	include_history : bool
		Whether to include history effects.

	Returns
	-------
	list
		A list containing the particle positions at each period, the initial
		position, and whether history effects were included.
	"""
	xdot_0, zdot_0 = system.flow.velocity(x_0, z_0, t=0)
	y = [x_0, z_0, xdot_0, zdot_0]
	x, z, _, _, _, _, _, _, _, _, _, _, _, _, \
	   _ = system.maxey_riley(t, y, include_history, HIDE_PROGRESS)
	return [x, z, x_0, z_0, include_history]

if __name__ == '__main__':
	main()

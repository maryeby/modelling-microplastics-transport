import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import repeat
from tqdm.contrib.itertools import product

from utils.data_tools import update_results
from utils.colors import print_failure
from transport_framework import particle as prt
from models import water_wave as fl
from models import my_system as ts
from examples.water_wave.horizontal_displacement.dibenedetto_numerics import \
	 DEPTH, AMPLITUDE, WAVELENGTH, SCALE
from examples.water_wave.horizontal_displacement.dibenedetto_numerics import \
	 STOKES_NUM, BETA

X_0S = np.round(np.linspace(0, 9 / (8 * np.pi), 5, endpoint=False), 5)
DB_ST = np.round(STOKES_NUM, 5)
DB_BETA = np.round(BETA, 5)
NUM_POINTS = 10
STOKES_NUMS = np.round(np.linspace(0.05, 0.1, NUM_POINTS), 3)
BETAS = np.round(np.linspace(0.95, 1.05, NUM_POINTS), 3)
DELTA_T = 5e-3
TOL = 5e-2
HIDE_PROGRESS = True
OUT_FILE = '../../data/water_wave/displacement_numerics.csv'

def main():
	"""
	Run numerical simulations for particles of varying densities and sizes.

	Simulations are run with and without history effects for particles of
	varying sizes (Stokes numbers) and densities as they are transported through
	a linear wave of arbitrarily deep water.
	"""
	# create dict to store solutions, ignore warnings
	results = {'x': [], 'z': [], 'x_0': [], 'St': [], 'beta': [], 'history': []}
	warnings.filterwarnings('ignore')

	# run varying Stokes number simulations
	for stokes_num, x_0 in product(STOKES_NUMS, X_0S):
		sols = run_numerics(x_0, stokes_num, DB_BETA, False)
		results = update_results(results, sols[:2], sols[2:])
		sols = run_numerics(x_0, stokes_num, DB_BETA, True)
		results = update_results(results, sols[:2], sols[2:])

	# run varying Stokes number simulations
	for beta, x_0 in product(BETAS, X_0S):
		sols = run_numerics(x_0, DB_ST, beta, False)
		results = update_results(results, sols[:2], sols[2:])
		sols = run_numerics(x_0, DB_ST, beta, True)
		results = update_results(results, sols[:2], sols[2:])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)

def estimate_num_periods(k, g, period, stokes_num, beta):
	"""
	Estimate the number of periods required for particles to reach the seabed.

	Parameters
	----------
	k : float
		The wavenumber *k'* associated with the flow.
	g : float
		The dimensionless gravity *g* acting on the particle.
	period : float
		The dimensionless wave period.
	"""
	h = -k * DEPTH
	r = beta * SCALE
	terminal_vel = stokes_num / r * (1 - beta) * g
	t_final = h / terminal_vel
	estimated_periods = t_final // period
	return int(5 * np.rint((np.abs(estimated_periods)) / 5)) + 5

def run_numerics(x_0, stokes_num, beta, include_history):
	""" 
	Run a numerical simulation with the specified *St* and density ratio *R*.

	Parameters
	----------
	x_0 : float
		The initial horizontal position of the particle.
	stokes_num : float
		The Stokes number to use for the initialization of the particle.
	beta : float
		The ratio between the particle and fluid densities.
	include_history : bool
		Whether history effects are included.

	Returns
	-------
	list
		A list containing the particle positions, *St*, and *R*.

	Notes
	-----
	A failure message is printed and the trajectory is plotted if the particle
	does not reach the seabed.
	"""
	# create Wave object, time series data, set initial particle position & vel
	wave = fl.WaterWave(DEPTH, AMPLITUDE, WAVELENGTH)
	z_0 = -1 if beta < 1 else -1.999
	xdot_0, zdot_0 = wave.velocity(x_0, z_0, t=0)
	y = [x_0, z_0, xdot_0, zdot_0]

	# estimate the number of periods necessary to reach the seabed/surface
	num_periods = estimate_num_periods(wave.wavenum, wave.gravity[1],
									   wave.period, stokes_num, beta)
	t = np.arange(0, num_periods * wave.period, DELTA_T)

	# compute R, create the Particle and TransportSystem objects
	density_ratio = beta * SCALE
	particle = prt.Particle(stokes_num)
	system = ts.MyTransportSystem(particle, wave, density_ratio)

	# run simulation and check if the particle reached the seabed
	x, z, _, _, _, _, _, _, _, _, _, _, _, _, \
	   _ = system.maxey_riley(t, y, include_history, HIDE_PROGRESS)
	reached_boundary = check_boundary(stokes_num, beta, include_history, z[-1],
									  wave.wavenum)
	if not reached_boundary:
		plt.figure()
		plt.title(rf'$\beta$ = {beta:g}, St = {stokes_num:g}')
		fmt = '--k.' if include_history else '-k.'
		plt.plot(x, z, fmt)
		plt.show()
	return [x, z, x_0, stokes_num, beta, include_history]

def check_boundary(stokes_num, beta, include_history, z_f, k):
	"""
	Print failure message if the particle has not reached the seabed or surface.

	Parameters
	----------
	stokes_num : float
		The Stokes number to use for the initialization of the particle.
	beta : float
		The ratio between the particle and fluid densities.
	include_history : bool
		Whether history effects are included.
	z_f : float
		The final vertical position of the particle.
	k : float
		The wavenumber *k'* associated with the flow.
	"""
	h = -k * DEPTH
	if beta < 1 and TOL < np.abs(h - z_f):
		print_failure('Particle did not reach the seabed.')
		print(f'\t {"seabed: ":<9}{h:>6.3f}')
		print(f'\t {"z_f: ":<9}{z_f:>6.3f}')
		print(f'\t {"St: ":<9}{stokes_num:>6.3f}')
		print(f'\t {"beta: ":<9}{beta:>6.3f}')
		print(f'\t {"history: ":<9} {include_history}')
		return False
	if beta > 1 and TOL < np.abs(z_f):
		print_failure('Particle did not reach the surface.')
		print(f'\t {"surface: ":<9}{0:>6.3f}')
		print(f'\t {"z_f: ":<9}{z_f:>6.3f}')
		print(f'\t {"St: ":<9}{stokes_num:>6.3f}')
		print(f'\t {"beta: ":<9}{beta:>6.3f}')
		print(f'\t {"history: ":<9} {include_history}')
		return False
	return True

if __name__ == '__main__': main()

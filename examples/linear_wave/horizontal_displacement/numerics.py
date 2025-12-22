import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import repeat
from tqdm.contrib.itertools import product

from utils.data_tools import update_results
from utils.colors import print_failure
from transport_framework import particle as prt
from models import linear_wave as fl
from models import my_system as ts
from examples.linear_wave.horizontal_displacement.dibenedetto_numerics import \
	 DEPTH, AMPLITUDE, WAVELENGTH
from examples.linear_wave.horizontal_displacement.dibenedetto_numerics import \
	 STOKES_HAT as DB_ST
from examples.linear_wave.horizontal_displacement.dibenedetto_numerics import \
	 R as DB_R

X_0S = np.round(np.linspace(0, 9 / (8 * np.pi), 5, endpoint=False), 5)
NUM_POINTS = 10
STOKES_HATS = np.round([0.48160, 0.53939, 0.58755, 0.64534, 0.69350, 0.75129,
						0.79945, 0.85724, 0.90540, 0.96319], 5)
RS = np.round([0.63333, 0.64067, 0.64800, 0.65533, 0.66267, 0.67067, 0.67800,
			   0.68533, 0.69267, 0.70000], 5)
DELTA_T = 5e-3
TOL = 5e-2
HIDE_PROGRESS = True
KEYS = ['x', 'z', 'x_0', 'Sthat', 'St', 'R', 'history']
OUT_FILE = '../../data/linear_wave/displacement_numerics.csv'

def main():
	"""
	Run numerical simulations for particles of varying densities and sizes.

	Simulations are run with and without history effects for particles of
	varying sizes (Stokes numbers) and densities as they are transported through
	a linear wave of arbitrarily deep water.
	"""
	# create dict to store solutions, ignore warnings
	results = {key: [] for key in KEYS}
	warnings.filterwarnings('ignore')

	# run varying Stokes number simulations
	for stokes_hat, x_0 in product(STOKES_HATS, X_0S):
		sols = run_numerics(x_0, stokes_hat, DB_R, False)
		results = update_results(results, sols[:2], sols[2:])
		sols = run_numerics(x_0, stokes_hat, DB_R, True)
		results = update_results(results, sols[:2], sols[2:])

	# run varying density ratio simulations
	for r, x_0 in product(RS, X_0S):
		sols = run_numerics(x_0, DB_ST, r, False)
		results = update_results(results, sols[:2], sols[2:])
		sols = run_numerics(x_0, DB_ST, r, True)
		results = update_results(results, sols[:2], sols[2:])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)

def estimate_num_periods(k, g, period, stokes_hat, r):
	"""
	Return the number of periods required for particles to reach the seabed.

	Parameters
	----------
	k : float
		The wavenumber *k'* associated with the flow.
	g : float
		The dimensionless gravity *g* acting on the particle.
	period : float
		The dimensionless wave period.
	stokes_hat : float
		The Stokes number to use for the initialization of the particle.
	r : float
		The ratio between the particle and fluid densities.
	"""
	h = -k * DEPTH
	terminal_vel = stokes_hat / r * ((1 - 3 * r / 2) / np.tanh(h))
	t_final = h / terminal_vel
	estimated_periods = t_final // period
	return int(5 * np.rint((np.abs(estimated_periods)) / 5)) + 5

def run_numerics(x_0, stokes_hat, r, include_history):
	r""" 
	Run a numerical simulation with the specified $\widehat{St}$ and $R$.

	Parameters
	----------
	x_0 : float
		The initial horizontal position of the particle.
	stokes_hat : float
		The Stokes number to use for the initialization of the particle.
	r : float
		The ratio between the particle and fluid densities.
	include_history : bool
		Whether history effects are included.

	Returns
	-------
	list
		A list containing the particle positions, $\widehat{St}$, $St$, and $R$.

	Notes
	-----
	A failure message is printed and the trajectory is plotted if the particle
	does not reach the seabed.
	"""
	# create Wave object, time series data, set initial particle position & vel
	wave = fl.LinearWave(DEPTH, AMPLITUDE, WAVELENGTH)
	z_0 = -1 if r < 2 / 3 else -1.999
	xdot_0, zdot_0 = wave.velocity(x_0, z_0, t=0)
	y = [x_0, z_0, xdot_0, zdot_0]

	# estimate the number of periods necessary to reach the seabed/surface
	num_periods = estimate_num_periods(wave.wavenum, wave.gravity[1],
									   wave.period, stokes_hat, r)
	t = np.arange(0, num_periods * wave.period, DELTA_T)
	if 20 < num_periods:
		print_warning('Large timespan, simulation(s) run with history effects '
					+ 'may be killed.')

	# compute R, create the Particle and TransportSystem objects
	particle = prt.Particle(stokes_hat)
	system = ts.MyTransportSystem(particle, wave, r)

	# run simulation and check if the particle reached the seabed
	x, z = system.maxey_riley(t, y, include_history, HIDE_PROGRESS)[:2]
	reached_boundary = check_boundary(stokes_hat, r, include_history, z[-1],
									  wave.wavenum)
	if not reached_boundary:
		plt.figure()
		plt.title(rf'$R$ = {r:g}, St = {system.stokes_num:g}')
		fmt = '--k.' if include_history else '-k.'
		plt.plot(x, z, fmt)
		plt.show()
	return [x, z, x_0, stokes_hat, system.stokes_num, r, include_history]

def check_boundary(stokes_hat, r, include_history, z_f, k):
	"""
	Print failure message if the particle has not reached the seabed or surface.

	Parameters
	----------
	stokes_hat : float
		The Stokes number to use for the initialization of the particle.
	r : float
		The ratio between the particle and fluid densities.
	include_history : bool
		Whether history effects are included.
	z_f : float
		The final vertical position of the particle.
	k : float
		The wavenumber *k'* associated with the flow.
	"""
	h = -k * DEPTH
	if r < 2 / 3 and TOL < np.abs(h - z_f):
		print_failure('Particle did not reach the seabed.')
		print(f'\t {"seabed: ":<9}{h:>6.3f}')
		print(f'\t {"z_f: ":<9}{z_f:>6.3f}')
		print(f'\t {"Sthat: ":<9}{stokes_hat:>6.3f}')
		print(f'\t {"R: ":<9}{r:>6.3f}')
		print(f'\t {"history: ":<9} {include_history}')
		return False
	if r > 2 / 3 and TOL < np.abs(z_f):
		print_failure('Particle did not reach the surface.')
		print(f'\t {"surface: ":<9}{0:>6.3f}')
		print(f'\t {"z_f: ":<9}{z_f:>6.3f}')
		print(f'\t {"Sthat: ":<9}{stokes_hat:>6.3f}')
		print(f'\t {"R: ":<9}{r:>6.3f}')
		print(f'\t {"history: ":<9} {include_history}')
		return False
	return True

if __name__ == '__main__': main()

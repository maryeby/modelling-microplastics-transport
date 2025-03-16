import warnings
import numpy as np
import pandas as pd
import itertools
from parallelbar import progress_starmap

from transport_framework import particle as prt
from models import deep_water_wave as fl
from models import my_system as ts

# wave conditions
AMPLITUDE = 0.02
WAVELENGTH = 1 
DEPTH = 10

# particle conditions
STOKES_NUMS = [0.01, 0.1, 1, 10]
NUM_POINTS = 4  # number of different initial vertical particle positions (z_0s)
X_0 = 0			# initial horizontal particle position

# simulation conditions
NUM_TASKS = len(STOKES_NUMS) * NUM_POINTS
R = 2 / 3		# denisty ratio
DELTA_T = 1e-3  # timestep
NUM_PERIODS = 3
INCLUDE_HISTORY = False
HIDE_PROGRESS = True
OUT_FILE = '../data/deep_water_wave/drift_velocity_numerics.csv'

def main():
	r"""
	Compute the average horizontal drift velocity of a particle in a wave.

	The Stokes drift velocity is numerically computed for neutrally buoyant
	particles with various Stokes numbers in a linear wave of infinitely deep
	water without history effects. Results are saved to the
	`data/deep_water_wave` directory.

	See Also
	--------
	models.my_system.compute_drift_velocity
	"""
	# create Wave object and array of timesteps
	wave = fl.DeepWaterWave(AMPLITUDE, WAVELENGTH, DEPTH)
	t = np.arange(0, NUM_PERIODS * wave.period, DELTA_T)
	z_0s = np.linspace(0, -wave.wavenum * DEPTH, NUM_POINTS, endpoint=False)

	# run numerics in parallel for each Stokes number and initial depth
	warnings.filterwarnings('ignore')
	paired_St, paired_z_0 = zip(*itertools.product(STOKES_NUMS, z_0s))
	params = zip(itertools.repeat(wave), paired_St, paired_z_0,
				 itertools.repeat(t))
	results = progress_starmap(run_numerics, params, n_cpu=4, total=NUM_TASKS)
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

def run_numerics(wave, stokes_num, z_0, t):
	"""
	Run a numerical simulation and compute the average drift velocity.

	Parameters
	----------
	wave : Wave (obj)
		The wave through which the particle is transported.
	stokes_num : float
		The Stokes number to use for the initialization of the particle.
	z_0 : float
		The initial vertical position of the particle.
	t : ndarray
		1D array containing `float` time series data.
	
	Returns
	-------
	dict
		Dictionary containing `float` data, the normalized average vertical
		particle position, the averaged horizontal Stokes drift velocity, and
		the Stokes number.
	"""
	# set the initial position and velocity of the particle
	xdot_0, zdot_0 = wave.velocity(X_0, z_0, t=0)
	y = [X_0, z_0, xdot_0, zdot_0]

	# initialize the particle and transport system
	particle = prt.Particle(stokes_num)
	system = ts.MyTransportSystem(particle, wave, R)

	# run simulation
	x, z, xdot, _, t, _, _, _, _, _, _, _, _, _, \
	   _ = system.maxey_riley(t, y, INCLUDE_HISTORY, HIDE_PROGRESS)

	# compute averaged horizontal drift velocity and scale results
	_, z_crossings, u, _, _ = ts.compute_drift_velocity(x, z, xdot, t)
	u_bar = np.mean(u) / wave.froude_num
	normalized_z_bar = np.mean(z_crossings) / (wave.wavenum * DEPTH)
	return {'z_bar/h': normalized_z_bar, 'u_bar': u_bar, 'St': stokes_num}

if __name__ == '__main__':
	main()

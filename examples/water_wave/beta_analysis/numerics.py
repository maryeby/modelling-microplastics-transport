import warnings
import numpy as np
import pandas as pd
from tqdm.contrib.itertools import product

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import water_wave as fl
from models import my_system as ts

# particle conditions
STOKES_NUM = 0.01
X_0 = 0

# wave conditions
DEPTH = 10
AMPLITUDE = 0.02
WAVELENGTH = 1

# simulation conditions
SCALE = 2 / 3
BETAS = [1, 0.8, 0.5]
R = SCALE * np.array(BETAS)
NUM_PERIODS = 20
DELTA_T = 5e-3
HIDE_PROGRESS = True
OUT_FILE = '../../data/water_wave/beta_numerics.csv'

def main():
	"""
	Run numerical simulations for particles of varying buoyancy in a wave.

	Simulations are performed for neutrally and negatively buoyant particles in
	linear waves of deep water, with and without history effects. Results are
	saved to the `data/water_wave` directory.
	"""
	# create the Particle, Flow, and TransportSystem objects
	particle = prt.Particle(STOKES_NUM)
	wave = fl.WaterWave(DEPTH, AMPLITUDE, WAVELENGTH)
	results = {'z_crossings': [], 'u_bar': [], 'beta': [], 'history': []}
	t = np.arange(0, NUM_PERIODS, DELTA_T)

	# neutrally buoyant case
	z_0s = np.linspace(-0.25, -4, 10, endpoint=False)
	z_0s = np.insert(z_0s, 0, -0.02)
	warnings.filterwarnings('ignore')
	for z_0, history in product(z_0s, [True, False]):
		# initialize variables for the simulation
		system = ts.MyTransportSystem(particle, wave, R[0])
		xdot_0, zdot_0 = wave.velocity(X_0, z_0, t=0)
		y = [X_0, z_0, xdot_0, zdot_0]

		# run simulation, compute drift velocity, and store solutions
		x, z, xdot, _, t, _, _, _, _, _, _, _, _, _, \
		   _ = system.maxey_riley(t, y, history, HIDE_PROGRESS)
		_, z_crossings, u, _, _ = ts.compute_drift_velocity(x, z, xdot, t)
		z = np.mean(z_crossings)
		u = np.mean(u)
		results = update_results(results, [], [z, u, BETAS[0], history])
		
	# negatively buoyant cases
	z_0 = 0
	for beta, history in product(BETAS[1:], [True, False]):
		b = BETAS.index(beta)
		system = ts.MyTransportSystem(particle, wave, R[b])
		xdot_0, zdot_0 = wave.velocity(X_0, z_0, t=0)
		y = [X_0, z_0, xdot_0, zdot_0]

		# run simulation, compute drift velocity, and store solutions
		x, z, xdot, _, t, _, _, _, _, _, _, _, _, _, \
		   _ = system.maxey_riley(t, y, history, HIDE_PROGRESS)
		_, z_crossings, u, _, _ = ts.compute_drift_velocity(x, z, xdot, t)
		results = update_results(results, [z_crossings[1:], u], [beta, history])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)

if __name__ == '__main__':
	main()

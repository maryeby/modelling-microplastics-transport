import warnings
import numpy as np
import pandas as pd

from transport_framework import particle as prt		# Particle class
from models import stokes_wave as sfl				# Flow (Wave) class
from models import water_wave as lfl				# Flow (Wave) class
from models import my_system as ts					# TransportSystem class
from utils.data_tools import update_results

# wave conditions
DEPTH = 10
AMPLITUDE = 0.0025
WAVELENGTH = 1.5

# particle conditions
STOKES_NUM = 0.01
X_0, Z_0 = 0, 0			# initial particle position

# simulation conditions
SCALE = 2 / 3			# to translate beta to R
BETA = 0.99				# density ratio
R = SCALE * BETA		# density ratio
NUM_PERIODS = 10
DELTA_T = 5e-3			# timestep size (recommended <= 5e-3)
INCLUDE_HISTORY = False
KEYS = ['t', 'x', 'z', 'xdot', 'zdot', 'u', 'w', 'wave']
OUT_FILE = '../data/stokes_wave/basic_numerics.csv'

def main():
	"""Run a numerical simulation of a particle transported through a wave."""
	# create the Particle, Flow, and TransportSystem objects
	particle = prt.Particle(STOKES_NUM)
	stokes_wave = sfl.StokesWave(DEPTH, AMPLITUDE, WAVELENGTH)
	linear_wave = lfl.WaterWave(DEPTH, AMPLITUDE, WAVELENGTH)
	results = {key: [] for key in KEYS}
	warnings.filterwarnings('ignore')

	# initialize and run simulation
	for wave in [linear_wave, stokes_wave]:
		name = 'linear' if wave == linear_wave else 'stokes'
		system = ts.MyTransportSystem(particle, wave, R)
		t = np.arange(0, NUM_PERIODS * wave.period, DELTA_T)
		xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
		y = [X_0, Z_0, xdot_0, zdot_0]	# initial particle position and velocity
		x, z, xdot, zdot = system.maxey_riley(t, y, INCLUDE_HISTORY)[:4]
		u, w = wave.velocity(x, z, t)
		results = update_results(results, [t, x, z, xdot, zdot, u, w], [name])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)

if __name__ == '__main__': main()

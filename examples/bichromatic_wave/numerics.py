import warnings
import numpy as np
import pandas as pd

from transport_framework import particle as prt		# Particle class
from models import bichromatic_wave as fl			# Flow (Wave) class
from models import my_system as ts					# TransportSystem class

# wave conditions
DEPTH = 10
AMPLITUDES = np.array([0.02, 0.025])
WAVELENGTHS = np.array([1, 1.5])

# particle conditions
STOKES_NUM = 0.01
X_0, Z_0 = 0, 0			# initial particle position

# simulation conditions
SCALE = 2 / 3			# to translate beta to R
BETA = 1				# density ratio
R = SCALE * BETA		# density ratio
NUM_PERIODS = 10
DELTA_T = 5e-3			# timestep size (recommended <= 5e-3)
INCLUDE_HISTORY = False
OUT_FILE = '../data/bichromatic_wave/basic_numerics.csv'

def main():
	"""Run a numerical simulation of a particle transported through a wave."""
	# create the Particle, Flow, and TransportSystem objects
	particle = prt.Particle(STOKES_NUM)
	wave = fl.BichromaticWave(DEPTH, AMPLITUDES, WAVELENGTHS)
	system = ts.MyTransportSystem(particle, wave, R)

	# initialize time series and set initial particle velocity = fluid velocity
	t = np.arange(0, NUM_PERIODS * wave.period[0], DELTA_T)
	xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
	y = [X_0, Z_0, xdot_0, zdot_0]	# initial particle position and velocity
	warnings.filterwarnings('ignore')

	# run simulation
	x, z, xdot, zdot, t, fpg_x, fpg_z, buoyancy_x, buoyancy_z, added_mass_x, \
	   added_mass_z, stokes_drag_x, stokes_drag_z, history_x, \
	   history_z = system.maxey_riley(t, y, INCLUDE_HISTORY)

	# store results in a dictionary and write the dictionary to a csv file
	results = {'t': t, 'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot}
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)

if __name__ == '__main__': main()

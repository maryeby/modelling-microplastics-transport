import warnings
import numpy as np
import pandas as pd

from transport_framework import particle as prt	# Particle class
from models import linear_wave as fl			# Flow (Wave) class
from models import my_system as ts				# TransportSystem class
from utils.data_tools import print_characteristic_params, update_results

# wave conditions
DEPTH = 10
AMPLITUDE = 0.02
WAVELENGTH = 1.5

# particle conditions
STOKES_HAT = np.round(19 / 205, 5)	# St = 0.1
X_0, Z_0 = 0, 0						# initial particle position

# simulation conditions
R = np.round(19 / 30, 5)			# density ratio
NUM_PERIODS = 5
DELTA_T = 5e-3						# timestep size (recommended <= 5e-3)
INCLUDE_HISTORY = True
OUT_FILE = '../../data/linear_wave/basic_numerics.csv'

def main():
	"""Run a simulation of a particle transported through a linear wave."""
	# create the Particle, Flow, and TransportSystem objects
	particle = prt.Particle(STOKES_HAT)
	wave = fl.LinearWave(DEPTH, AMPLITUDE, WAVELENGTH)
	system = ts.MyTransportSystem(particle, wave, R)
	print_characteristic_params(wave, particle, system)

	# initialize time series and set initial particle velocity = fluid velocity
	t = np.arange(0, NUM_PERIODS * wave.period, DELTA_T)
	xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
	y = [X_0, Z_0, xdot_0, zdot_0]	# initial particle position and velocity
	results = {'t': [], 'x': [], 'z': [], 'xdot': [], 'zdot': []}
	warnings.filterwarnings('ignore')

	# run simulation and compute drift velocity
	x, z, xdot, zdot, t, fpg_x, fpg_z, buoyancy_x, buoyancy_z, \
		added_mass_x, added_mass_z, stokes_drag_x, stokes_drag_z, \
		history_x, history_z = system.maxey_riley(t, y, INCLUDE_HISTORY)
	results = update_results(results, [t, x, z, xdot, zdot], [])
	x_cross, z_cross, _, _, _ = ts.compute_drift_velocity(x, z, xdot, t)

	# store results in a dictionary and write the dictionary to a csv file
	print_characteristic_params(wave, particle, system)
	results = {'t': t, 'x': x, 'z': z, 'xdot': xdot, 'zdot': zdot,
			   'x_crossings': x_cross, 'z_crossings': z_cross}
	pd.DataFrame(dict([(key, pd.Series(value)) for key, value 
				 in results.items()])).to_csv(OUT_FILE, index=False)

if __name__ == '__main__': main()

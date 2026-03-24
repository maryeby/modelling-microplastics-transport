import warnings
import numpy as np
import pandas as pd

from transport_framework import particle as prt	# Particle class
from models import bichromatic_wave as fl		# Flow (Wave) class
from models import my_system as ts				# TransportSystem class
from utils.data_tools import update_results, print_characteristic_params

# wave conditions
DEPTH = 2
AMPLITUDES = np.array([0.3, 0.2])
WAVELENGTHS = np.array([36, 50])
SLOPE = -0.0095

# simulation conditions
R = 0.666							# density ratio
NUM_PERIODS = 5
DELTA_T = 5e-3						# timestep size (recommended <= 5e-3)

# particle conditions
STOKES_HAT = 0.2
X_0, Z_0 = 0, -0.15					# initial particle position
OUT_FILE = '../../data/bichromatic_wave/basic_numerics.csv'

def main():
	"""Run a simulation of a particle transported through a bichromatic wave."""
	# create the Particle, Flow, and TransportSystem objects
	warnings.filterwarnings('ignore')
	particle = prt.Particle(STOKES_HAT)
	wave = fl.BichromaticWave(DEPTH, AMPLITUDES, WAVELENGTHS)
	system = ts.MyTransportSystem(particle, wave, R)
	print_characteristic_params(wave, particle, system)

	# initialize time series and set initial particle velocity = fluid velocity
	xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
	y = [X_0, Z_0, xdot_0, zdot_0]
	results = {'t': [], 'x': [], 'z': [], 'xdot': [], 'zdot': [], 'history': []}

	# run simulations
	for h in [False, True]:
		t = np.arange(0, NUM_PERIODS * wave.period[0], DELTA_T)
		x, z, xdot, zdot, t, fpg_x, fpg_z, buoyancy_x, buoyancy_z, \
			added_mass_x, added_mass_z, stokes_drag_x, stokes_drag_z, \
			history_x, history_z = system.maxey_riley(t, y, h)
		results = update_results(results, [t, x, z, xdot, zdot], [h])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)

if __name__ == '__main__': main()

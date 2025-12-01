import warnings
import numpy as np
import pandas as pd

from utils.data_tools import update_results, print_characteristic_params
from utils.colors import print_warning
from transport_framework import particle as prt
from models import linear_wave as fl
from models import my_system as ts

# wave conditions
DEPTH = 0.5
AMPLITUDE = 0.025
WAVELENGTH = 4e4 * np.pi / 83057

# simulation conditions
SCALE = 2 / 3
BETA = 75 / 76
R = np.round(SCALE * BETA, 5)
NUM_PERIODS = 16
DELTA_T = 1e-2

# particle conditions
X_0, Z_0 = 0, 0
STOKES_HAT = np.round(R * 2 * np.pi * 5e-4 * 5e-4 / (9 * 1e-6)
						* (1000 + 2 * 1020) / 1000, 5)
OUT_FILE = '../../data/linear_wave/dibenedetto_numerics.csv'

def main():
	"""
	Run numerical simulations for a negatively buoyant particle in a wave.

	Simulations are run with and without history effects for a negatively
	buoyant particle transported through a linear wave of arbitrarily deep
	water. Conditions for the simulations were chosen based on [1] Figure 8.

	References
	----------
	[^1]: [M. H. DiBenedetto et al. (2022).](
		  https://doi.org/10.1017/jfm.2022.95) Enhanced settling and dispersion
		  of inertial particles in surface waves. *Journal of Fluid Mechanics*
		  936, A38.
	"""
	# create the Particle, Flow, and TransportSystem objects
	particle = prt.Particle(STOKES_HAT)
	wave = fl.LinearWave(DEPTH, AMPLITUDE, WAVELENGTH)
	system = ts.MyTransportSystem(particle, wave, R)
	print_characteristic_params(particle, wave, system)
	if 20 < NUM_PERIODS:
		print_warning('Large timespan, simulation(s) run with history effects '
					+ 'may be killed.')

	# create dict to store sols, initialize variables for the simulations
	results = {'x': [], 'z': [], 'history': []}
	t = np.arange(0, NUM_PERIODS * wave.period, DELTA_T)
	xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
	x_f = []

	# run simulations with and without history effects
	warnings.filterwarnings('ignore')
	for history in [True, False]:
		x, z = system.maxey_riley(t, y, include_history=history)[:2]
		x_f.append(x[-1])
		results = update_results(results, [x, z], [history])

	# print the percent displacement and write to data file
	print(f'{x_f[0] * 100 / x_f[1] - 100:.2f}%')
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)

if __name__ == '__main__':
	main()

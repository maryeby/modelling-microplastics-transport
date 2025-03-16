import warnings
import numpy as np
import pandas as pd

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import water_wave as fl
from models import my_system as ts

# wave conditions
DEPTH = 0.5
AMPLITUDE = 0.025
WAVELENGTH = 4e4 * np.pi / 83057

# particle conditions
STOKES_NUM = 2 * AMPLITUDE * np.pi * np.pi / (9 * WAVELENGTH)
X_0, Z_0 = 0, 0

# simulation conditions
SCALE = 2 / 3
BETA = 75 / 76
R = SCALE * BETA
NUM_PERIODS = 50
DELTA_T = 1e-2
OUT_FILE = '../../data/water_wave/displacement_numerics.csv'

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
	particle = prt.Particle(STOKES_NUM)
	wave = fl.WaterWave(DEPTH, AMPLITUDE, WAVELENGTH)
	system = ts.MyTransportSystem(particle, wave, R)

	# create dict to store sols, initialize variables for the simulations
	results = {'x': [], 'z': [], 'history': []}
	xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
	t = np.arange(0, NUM_PERIODS * wave.period, DELTA_T)
	y = [X_0, Z_0, xdot_0, zdot_0]
	x_f = []

	# run simulations and store results
	warnings.filterwarnings('ignore')
	for history in [False, True]:
		x, z, _, _, _, _, _, _, _, _, _, _, _, _, \
		   _ = system.maxey_riley(t, y, include_history=history)
		x_f.append(x[-1])
		results = update_results(results, [x, z], [history])

	# print the percent displacement and write to data file
	print(f'{x_f[1] * 100 / x_f[0] - 100:.2f}%')
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)

if __name__ == '__main__':
	main()

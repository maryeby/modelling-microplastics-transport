import numpy as np
import pandas as pd

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import dim_deep_linear_wave as dfl
from models import deep_linear_wave as fl
from models import santamaria_system as sts
from models import my_system as ts

# wave conditions
AMPLITUDE = 0.026
WAVELENGTH = 0.5

# particle conditions
SCALE = 2 / 3					# for translating parameters
STOKES_NUM = 0.5				# St from Santamaria et al. (2013)
HEAVY_BETA = 0.96
LIGHT_BETA = 1.04
HEAVY_R = SCALE * HEAVY_BETA
LIGHT_R = SCALE * LIGHT_BETA
HEAVY_STHAT = SCALE * HEAVY_BETA * STOKES_NUM
LIGHT_STHAT = SCALE * LIGHT_BETA * STOKES_NUM

# simulation conditions
NUM_PERIODS = 40
DELTA_T = 1e-2
INCLUDE_HISTORY = False
OUT_FILE = '../../data/deep_linear_wave/santamaria_fig1_recreation.csv' 

def main():
	"""
	Reproduce numerical results from Figure 1 in [1].

	Simulations are run for a negatively buoyant particle and positively buoyant
	particle in a linear wave of infinitely deep water without history effects.
	Results are saved to the `data/deep_linear_wave` directory.

	References
	----------
	[^1]: [F. Santamaria et al. (2013).](
		  https://doi.org/10.1209/0295-5075/102/14003)
		  Stokes drift for inertial particles transported by water waves.
		  *EPL (Europhysics Letters)*, 102(1), 14003.
	"""
	results = {'x': [], 'z': [], 'beta': [], 'method': []}

	# heavy particle, numerical integration using Santamaria transport system
	particle = prt.Particle(HEAVY_STHAT)
	wave = dfl.DimensionalDeepLinearWave(AMPLITUDE, WAVELENGTH)
	system = sts.SantamariaTransportSystem(particle, wave, HEAVY_BETA)
	x_0, z_0 = 0, 0
	timestep = DELTA_T / wave.angular_freq # scale time
	print('Running Santamaria simulation for the heavy particle...', end='')
	x, z, _, _, _ = system.run_numerics(system.maxey_riley, x_0, z_0,
										NUM_PERIODS, timestep)
	print('done.')
	x *= wave.wavenum
	z *= wave.wavenum
	results = update_results(results, [x, z], [HEAVY_BETA, 'Santamaria'])

	# light particle, numerical integration using Santamaria transport system
	particle = prt.Particle(LIGHT_STHAT)
	system = sts.SantamariaTransportSystem(particle, wave, LIGHT_BETA)
	x_0, z_0 = 0.13, -0.4
	print('Running Santamaria simulation for the light particle...', end='')
	x, z, _, _, _ = system.run_numerics(system.maxey_riley, x_0, z_0,
										NUM_PERIODS, timestep)
	print('done.')
	x *= wave.wavenum
	z *= wave.wavenum
	results = update_results(results, [x, z], [LIGHT_BETA, 'Santamaria'])

	# heavy particle, multi-step integration scheme using our model
	wave = fl.DeepLinearWave(AMPLITUDE, WAVELENGTH)
	particle = prt.Particle(STOKES_NUM * SCALE * HEAVY_BETA)
	system = ts.MyTransportSystem(particle, wave, HEAVY_R)
	x_0, z_0 = 0, 0
	xdot_0, zdot_0 = wave.velocity(x_0, z_0, t=0)
	y = [x_0, z_0, xdot_0, zdot_0]
	t = np.arange(0, NUM_PERIODS * wave.period, DELTA_T)
	x, z, _, _, _, _, _, _, _, _, _, _, _, _,\
		  _ = system.maxey_riley(t, y, INCLUDE_HISTORY)
	results = update_results(results, [x, z], [HEAVY_BETA, 'Daitche'])

	# light particle, multi-step integration scheme using our model
	particle = prt.Particle(STOKES_NUM * SCALE * LIGHT_BETA)
	system = ts.MyTransportSystem(particle, wave, LIGHT_R)
	x_0, z_0 = 0.13 * wave.wavenum, -0.4 * wave.wavenum
	xdot_0, zdot_0 = wave.velocity(x_0, z_0, t=0)
	y = [x_0, z_0, xdot_0, zdot_0]
	x, z, _, _, _, _, _, _, _, _, _, _, _, _,\
		  _ = system.maxey_riley(t, y, INCLUDE_HISTORY)
	results = update_results(results, [x, z], [LIGHT_BETA, 'Daitche'])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

if __name__ == '__main__':
	main()

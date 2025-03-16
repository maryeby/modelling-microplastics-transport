import numpy as np
import pandas as pd

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import dim_deep_water_wave as dfl
from models import deep_water_wave as fl
from models import santamaria_system as sts
from models import my_system as ts

# wave conditions
AMPLITUDE = 0.02
WAVELENGTH = 1

# particle conditions
X_0, Z_0 = 0, 0
STOKES_NUM = 0.157

# simulation conditions
SCALE = 2 / 3
BETA = 0.9
R = SCALE * BETA
NUM_PERIODS = 50
DELTA_TS = [1e-3, 5e-3, 1e-2]
INCLUDE_HISTORY = False
OUT_FILE = '../../data/deep_water_wave/santamaria_fig2_recreation.csv'

def main():
	"""
	Reproduce numerical results from Figure 2 in [1].

    Simulations are run for a negatively buoyant particle in a linear wave of
	infinitely deep water without history effects. Results are saved to the
	`data/deep_water_wave` directory.

    References
    ----------
    [^1]: [F. Santamaria et al. (2013).](
		  https://doi.org/10.1209/0295-5075/102/14003)
          Stokes drift for inertial particles transported by water waves.
          *EPL (Europhysics Letters)*, 102(1), 14003.
	"""
	# create dictionary to store solutions and variables for simulation
	results = {'t': [], 'u_bar': [], 'w_bar': [], 'delta_t': [], 'method': []}
	particle = prt.Particle(STOKES_NUM)
	wave = dfl.DimensionalDeepWaterWave(AMPLITUDE, WAVELENGTH)
	system = sts.SantamariaTransportSystem(particle, wave, BETA)
	delta_t = DELTA_TS[0] / (wave.angular_freq * wave.froude_num)

	# numerically integrate
	x, z, xdot, _, t = system.run_numerics(system.maxey_riley, X_0, Z_0,
										   NUM_PERIODS, delta_t)
	# scale solutions
	x *= wave.wavenum
	z *= wave.wavenum
	xdot /= wave.max_velocity
	t *= wave.angular_freq * wave.froude_num

	# compute drift velocity and store solutions
	_, _, u_bar, w_bar, t = ts.compute_drift_velocity(x, z, xdot, t)
	results = update_results(results, [t[1:], u_bar, w_bar], [DELTA_TS[0],
							'Santamaria'])

	# create DeepWaterWave, Particle, and TransportSystem objects
	wave = fl.DeepWaterWave(AMPLITUDE, WAVELENGTH)
	particle = prt.Particle(STOKES_NUM * wave.froude_num * R)
	system = ts.MyTransportSystem(particle, wave, R)

	# compute numerics for various delta_ts and store solutions
	for delta_t in DELTA_TS: compute_numerics(system, delta_t, results)
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

def compute_numerics(system, delta_t, results):
	"""
	Run simulation and compute the numerical Stokes drift velocity.

	Parameters
	----------
	system : TransportSystem (obj)
		The transport system through which the simulation is run.
	delta_t : float
		The timestep size.
	results : dict
		The dictionary the solutions are stored in.

	See Also
	--------
	models.my_system.compute_drift_velocity
	"""
	xdot_0, zdot_0 = system.flow.velocity(X_0, Z_0, t=0)
	y = [X_0, Z_0, xdot_0, zdot_0]
	t = np.arange(0, NUM_PERIODS * system.flow.period, delta_t)
	x, z, xdot, _, t, _, _, _, _, _, _, _, _, _, \
	   _ = system.maxey_riley(t, y, INCLUDE_HISTORY)
	_, _, u_bar, w_bar, t = ts.compute_drift_velocity(x, z, xdot, t)
	results = update_results(results, [t[1:], u_bar, w_bar],
							[delta_t, 'Daitche'])

if __name__ == '__main__':
	main()

import pandas as pd
import numpy as np
import scipy.constants as constants

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import dim_deep_linear_wave as fl
from models import santamaria_system as ts
from examples.deep_linear_wave.santamaria.fig2_numerics import AMPLITUDE, \
	 WAVELENGTH, X_0, Z_0, STOKES_HAT, BETA, NUM_PERIODS, DELTA_TS

OUT_FILE = '../../data/deep_linear_wave/santamaria_analytics.csv'

def main():
	"""
	Compute analytical solutions for the drift velocity of a particle in a wave.
	
	The Stokes drift velocity is analytically computed[^1] for a particle in a
	linear wave of infinitely deep water for the purposes of recreating Figure 2
	from Ref. 1. Equations (13) and (14) from Ref. 1 are implemented here with
	slight modifications. Results are saved to the `data/deep_linear_wave`
	directory.

	References
	----------
	[^1]: [F. Santamaria et al. (2013).](
		  https://doi.org/10.1209/0295-5075/102/14003)
		  Stokes drift for inertial particles transported by water waves.
		  *EPL (Europhysics Letters)*, 102(1), 14003.
	"""
	# initialize the objects and related parameters
	particle = prt.Particle(STOKES_HAT)
	wave = fl.DimensionalDeepLinearWave(AMPLITUDE, WAVELENGTH)
	system = ts.SantamariaTransportSystem(particle, wave, BETA)
	stokes_num = system.stokes_num
	k = wave.wavenum
	omega = wave.angular_freq
	u = wave.max_velocity
	c = wave.phase_velocity

	# initialize the remaining variables for computing the analytics
	results = {'t': [], 'u_d': [], 'w_d': [], 'settling_velocity': []}
	delta_t = DELTA_TS[0]
	t = np.arange(0, NUM_PERIODS * wave.period, delta_t)
	bprime = 1 - BETA
	e_2z0t = np.exp(2 * (k * Z_0 - stokes_num * bprime * t * omega))

	# compute analytical drift velocity and settling velocity
	u_d = u ** 2 / c * e_2z0t * (1 - stokes_num ** 2 * bprime)
	w_d = -c * stokes_num * bprime * (1 + 2 * (u / c) ** 2 * e_2z0t)
	settling_velocity = -bprime * constants.g * (stokes_num / omega)

	# scale and store results
	t *= omega
	u_d /= u
	w_d /= u
	settling_velocity /= u
	results = update_results(results, [t, u_d, w_d], [settling_velocity])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)

if __name__ == '__main__':
	main()

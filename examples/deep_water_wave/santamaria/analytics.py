import pandas as pd
import numpy as np
import scipy.constants as constants

from utils.data_tools import update_results
from models import dim_deep_water_wave as fl
from examples.deep_water_wave.santamaria.fig2_numerics import AMPLITUDE, \
	 WAVELENGTH, X_0, Z_0, STOKES_NUM, BETA, NUM_PERIODS, DELTA_TS

OUT_FILE = '../../data/deep_water_wave/santamaria_analytics.csv'

def main():
	"""
	Compute analytical solutions for the drift velocity of a particle in a wave.
	
	The Stokes drift velocity is analytically computed[^1] for a particle in a
	linear wave of infinitely deep water for the purposes of recreating Figure 2
	from [1]. Equations (13) and (14) from [1] are implemented here with slight
	modifications. Results are saved to the `data/deep_water_wave` directory.

	References
	----------
	[^1]: [F. Santamaria et al. (2013).](
		  https://doi.org/10.1209/0295-5075/102/14003)
		  Stokes drift for inertial particles transported by water waves.
		  *EPL (Europhysics Letters)*, 102(1), 14003.
	"""
	# initialize the flow and related parameters
	wave = fl.DimensionalDeepWaterWave(AMPLITUDE, WAVELENGTH)
	k = wave.wavenum
	omega = wave.angular_freq
	U = wave.max_velocity
	c = wave.phase_velocity

	# initialize the remaining variables for computing the analytics
	results = {'t': [], 'u_d': [], 'w_d': [], 'settling_velocity': []}
	delta_t = DELTA_TS[0] / (omega * wave.froude_num)
	t = np.arange(0, NUM_PERIODS * wave.period, delta_t)
	bprime = 1 - BETA
	e_2z0t = np.exp(2 * (k * Z_0 - STOKES_NUM * bprime * t * omega))

	# compute analytical drift velocity and settling velocity
	u_d = U ** 2 / c * e_2z0t * (1 - STOKES_NUM ** 2 * bprime)
	w_d = -c * STOKES_NUM * bprime * (1 + 2 * (U / c) ** 2 * e_2z0t)
	settling_velocity = -bprime * constants.g * (STOKES_NUM / omega)

	# scale and store results
	t *= omega * wave.froude_num
	u_d /= U
	w_d /= U
	settling_velocity /= U
	results = update_results(results, [t, u_d, w_d], [settling_velocity])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False)

if __name__ == '__main__':
	main()

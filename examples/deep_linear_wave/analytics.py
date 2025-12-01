import pandas as pd
import numpy as np

from models import deep_linear_wave as fl
from examples.deep_linear_wave.drift_velocity_numerics import AMPLITUDE, \
															  WAVELENGTH, DEPTH
OUT_FILE = '../data/deep_linear_wave/analytics.csv'

def main():
	r"""
	Compute analytical solutions for the drift velocity of a particle in a wave.
	
	The dimensionless horizontal Stokes drift velocity is computed[^1] for a
	neutrally buoyant particle in a linear wave of infinitely deep water.
	Results are saved to the `data/deep_linear_wave` directory.

	Notes
	-----
	The non-dimensional equation, $$u_d = \epsilon^2 e^{2z},$$ is based on the
	dimensional equation[^1], $$u'_{SD} = c'(A'k')^2 e^{2k'z'}.$$

	References
	----------
	[^1]: [T. S. van den Bremer & Ø. Breivik (2018).](
		  https://doi.org/10.1098/rsta.2017.0104) Stokes drift.
		  *Philosophical Transactions of the Royal Society A: Mathematical,
		  Physical and Engineering Sciences* 376(2111), 20170104.
	"""
	# initialize the flow (wave) and related parameters
	wave = fl.DeepLinearWave(AMPLITUDE, WAVELENGTH, DEPTH)
	epsilon = wave.steepness
	k = wave.wavenum
	h = k * DEPTH
	z = np.linspace(0, -h, 100)

	# compute normalized drift velocity and write results to data file
	u_d = np.exp(2 * z)
	pd.DataFrame({'z/h': z / h, 'u_d': u_d}).to_csv(OUT_FILE, index=False)

if __name__ == '__main__':
	main()

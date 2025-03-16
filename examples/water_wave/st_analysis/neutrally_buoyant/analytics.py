import pandas as pd
import numpy as np

from utils.data_tools import update_results
from models import water_wave as fl
from examples.water_wave.st_analysis.neutrally_buoyant.numerics import DEPTHS, \
	 AMPLITUDE, WAVELENGTH

OUT_FILE = '../../../data/water_wave/st_analytics.csv'

def main():
	r"""
	Compute analytical solutions for the horizontal Stokes drift velocity.[^1]

	The non-dimensional horizontal Stokes drift velocity is computed for
	neutrally buoyant particles in waves of arbitrarily deep water. Results are
	saved to the `data/deep_water_wave` directory.

	Notes
	-----
	The computation is performed using the expression,
	$$u_d = \frac{\cosh{(2(z + h))}}{2\sinh^2(h)},$$
	based on the dimensional equation,[^1]
	$$u'_{SD} = c'(A'k')^2 \frac{\cosh{(2k'(z' + h'))}}{2\sinh^2(k'h')}.$$

	References
	----------
	[^1]: [T. S. van den Bremer & Ø. Breivik (2018).](
		  https://doi.org/10.1098/rsta.2017.0104) Stokes drift.
		  *Philosophical Transactions of the Royal Society A: Mathematical,
		  Physical and Engineering Sciences* 376(2111), 20170104.
	"""
	results = {'z/h': [], 'u_d': [], 'depth': []}
	for depth in DEPTHS:
		# initialize the Wave object and related parameters
		wave = fl.WaterWave(depth, AMPLITUDE, WAVELENGTH)
		k = wave.wavenum
		h = k * depth

		# compute drift velocity and store solutions
		z = np.linspace(0, -h, 100)
		u_d = np.cosh(2 * (z + h)) / (2 * np.sinh(h) ** 2)
		z /= h
		results = update_results(results, [z, u_d], [depth])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

if __name__ == '__main__':
	main()
